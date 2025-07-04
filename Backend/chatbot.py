import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import ollama
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import logging
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
class Config:
    OLLAMA_MODEL = "gemma2:2b"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key")
    PINECONE_ENV = "us-east-1"
    INDEX_NAME = "user-preferences"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="Personalized Chatbot API",
    description="API for a chatbot that remembers user preferences using Pinecone",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - specify exact origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Components (Initialized at startup) ---
pinecone_index = None
embedder = None

# --- Models ---
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    processing_time_ms: float
    status: str = "success"

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize all required components at startup."""
    global pinecone_index, embedder
    
    # Initialize Pinecone
    pinecone_start = time.time()
    try:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        if Config.INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=Config.INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-2")
            )
            logger.info(f"Created new Pinecone index: {Config.INDEX_NAME}")
        
        pinecone_index = pc.Index(Config.INDEX_NAME)
        logger.info(f"✅ Pinecone initialized in {(time.time() - pinecone_start):.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Pinecone initialization failed")

    # Initialize Embedding Model
    embedder_start = time.time()
    try:
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        logger.info(f"✅ Embedding model loaded in {(time.time() - embedder_start):.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise HTTPException(status_code=500, detail="Embedding model loading failed")

    # Warm up models with a test query
    try:
        test_start = time.time()
        ollama.chat(model=Config.OLLAMA_MODEL, messages=[{"role": "user", "content": "Hello"}])
        logger.info(f"✅ Ollama model warmed up in {(time.time() - test_start):.2f}s")
    except Exception as e:
        logger.warning(f"⚠️ Ollama warmup failed (might still work): {e}")

# --- Helper Functions ---
def get_user_context(user_id: str) -> str:
    """Fetch stored user info from Pinecone."""
    try:
        user_data = pinecone_index.fetch(ids=[user_id])
        if user_id in user_data.vectors:
            stored_info = user_data.vectors[user_id].metadata.get("info", "")
            return stored_info if stored_info else "No specific user preferences found."
        return "No specific user preferences found."
    except Exception as e:
        logger.error(f"Error retrieving user context: {e}")
        return "Error retrieving user context."

def summarize_chat(user_query: str, bot_reply: str) -> str:
    """Generate a summary of the latest chat interaction."""
    prompt = f"Summarize this user-bot interaction briefly:\n\nUser: {user_query}\nBot: {bot_reply}\n\nSummary:"
    try:
        response = ollama.chat(
            model=Config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Summary could not be generated."

def store_user_info(user_id: str, new_info: str):
    """Stores or updates user preferences in Pinecone."""
    try:
        # Check if user already exists
        existing_data = pinecone_index.fetch(ids=[user_id])
        
        if user_id in existing_data.vectors:
            # User exists - update their info
            existing_info = existing_data.vectors[user_id].metadata.get("info", "")
            combined_info = f"{existing_info}\n{new_info}".strip()
        else:
            # New user - create fresh entry
            combined_info = new_info
        
        # Generate embedding for the combined info
        combined_embedding = embedder.encode(combined_info).tolist()
        
        # Store/update the user's info
        pinecone_index.upsert(
            vectors=[(user_id, combined_embedding, {"info": combined_info})]
        )
        
        logger.info(f"Updated preferences for user {user_id}")
    except Exception as e:
        logger.error(f"Error storing summary for user {user_id}: {e}")

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that handles user messages with context."""
    start_time = time.time()
    
    try:
        # Get user context
        context = get_user_context(request.user_id)
        
        # Prepare the prompt with context
        prompt = (
            f"Here is what I know about you (User {request.user_id}):\n{context}\n\n"
            f"Now, answer the user's question directly without introducing unrelated topics.\n\n"
            f"User Query: {request.message}\n\nResponse:"
        )
        
        # Generate response
        response_start = time.time()
        response = ollama.chat(
            model=Config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        bot_reply = response["message"]["content"]
        response_time = time.time() - response_start
        
        # Summarize and store the interaction
        summary_start = time.time()
        summary = summarize_chat(request.message, bot_reply)
        store_user_info(request.user_id, summary)
        storage_time = time.time() - summary_start
        
        logger.info(
            f"Chat processed - "
            f"Response: {response_time:.2f}s, "
            f"Storage: {storage_time:.2f}s, "
            f"Total: {time.time() - start_time:.2f}s"
        )
        
        return ChatResponse(
            response=bot_reply,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "response": f"Error processing your request: {str(e)}",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        )

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "pinecone_ready": pinecone_index is not None,
        "embedder_ready": embedder is not None
    }