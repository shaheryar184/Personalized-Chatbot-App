import os
import time
import ollama
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
OLLAMA_MODEL = "gemma2:2b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Efficient embeddings

# Load Pinecone API Key securely
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "user-preferences"

# --- INITIALIZE PINECONE ---
try:
    start_time = time.time()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # MiniLM embedding size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-2")
        )
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone successfully! (Time: {time.time() - start_time:.4f} sec)")
except Exception as e:
    print(f"❌ Error initializing Pinecone: {e}")
    exit()

# Load embedding model
start_time = time.time()
embedder = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Embedding model loaded! (Time: {time.time() - start_time:.4f} sec)")

def get_user_context(user_id):
    """Fetch stored user info from Pinecone for specific user only."""
    start_time = time.time()
    try:
        user_data = index.fetch(ids=[user_id])
        if user_id in user_data.vectors:
            stored_info = user_data.vectors[user_id].metadata.get("info", "")
            print(f"🕒 User context fetched in {time.time() - start_time:.4f} sec")
            return stored_info if stored_info else "No specific user preferences found."
        return "No specific user preferences found."
    except Exception as e:
        print(f"❌ Error retrieving user context: {e} (Time: {time.time() - start_time:.4f} sec)")
        return "Error retrieving user context."

def summarize_chat(user_query, bot_reply):
    """Generate a summary of the latest chat interaction."""
    prompt = f"Summarize this user-bot interaction briefly:\n\nUser: {user_query}\nBot: {bot_reply}\n\nSummary:"
    try:
        start_time = time.time()
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        summary = response["message"]["content"]
        print(f"🕒 Summary generated in {time.time() - start_time:.4f} sec")
        return summary
    except Exception as e:
        print(f"❌ Error generating summary: {e} (Time: {time.time() - start_time:.4f} sec)")
        return "Summary could not be generated."

def store_user_info(user_id, new_info):
    """Stores or updates user preferences in Pinecone for specific user only."""
    start_time = time.time()
    try:
        # Check if user already exists
        existing_data = index.fetch(ids=[user_id])
        
        if user_id in existing_data.vectors:
            # User exists - update their info
            existing_info = existing_data.vectors[user_id].metadata.get("info", "")
            combined_info = f"{existing_info}\n{new_info}".strip()
        else:
            # New user - create fresh entry
            combined_info = new_info
        
        # Generate embedding for the combined info
        embedding_start = time.time()
        combined_embedding = embedder.encode(combined_info).tolist()
        print(f"🕒 Embedding generated in {time.time() - embedding_start:.4f} sec")
        
        # Store/update the user's info
        upsert_start = time.time()
        index.upsert(vectors=[(user_id, combined_embedding, {"info": combined_info})])
        print(f"🕒 Pinecone upsert completed in {time.time() - upsert_start:.4f} sec")
        
        print(f"✅ User {user_id}'s summary updated successfully! (Total time: {time.time() - start_time:.4f} sec)")

    except Exception as e:
        print(f"❌ Error storing summary for user {user_id}: {e} (Time: {time.time() - start_time:.4f} sec)")

def chat():
    """Runs the chatbot with strict user-specific history tracking."""
    print("\nChatbot is running! Type 'exit' to stop.")
    
    # Get user ID (in a real app, this would come from authentication)
    user_id = input("Please enter your user ID: ").strip()
    if not user_id:
        user_id = "default_user"
        print(f"⚠️  No user ID provided, using '{user_id}'")
    
    messages = []  # Stores conversation history

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Get context specific to this user only
        context_start = time.time()
        context = get_user_context(user_id)
        print(f"🕒 Total context retrieval time: {time.time() - context_start:.4f} sec")
        
        messages.append({"role": "user", "content": user_query})
        
        prompt = f"Here is what I know about you (User {user_id}):\n{context}\n\nNow, answer the user's question directly without introducing unrelated topics.\n\nUser Query: {user_query}\n\nResponse:"

        try:
            response_start = time.time()
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            bot_reply = response["message"]["content"]
            print(f"🕒 Response generated in {time.time() - response_start:.4f} sec")
        except Exception as e:
            bot_reply = f"❌ Error generating response: {e}"
            print(f"🕒 Error during response generation: {time.time() - response_start:.4f} sec")

        print("\nBot:", bot_reply)

        # Summarize the interaction and store it for this user only
        summary_start = time.time()
        summary = summarize_chat(user_query, bot_reply)
        store_user_info(user_id, summary)
        print(f"🕒 Total summary and storage time: {time.time() - summary_start:.4f} sec")

        messages.append({"role": "assistant", "content": bot_reply})
        if len(messages) > 10:
            messages = messages[-10:]

if __name__ == "__main__":
    chat()