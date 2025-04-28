import { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  Modal,
  Pressable,
  Image,
  FlatList,
  Dimensions,
} from 'react-native';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';

// Mock data for chat history
const CHAT_HISTORY = [
  { id: '1', title: 'Project Overview', timestamp: '2 hours ago' },
  { id: '2', title: 'Code Optimization', timestamp: 'Yesterday' },
  { id: '3', title: 'React Native Help', timestamp: '2 days ago' },
  { id: '4', title: 'App Architecture', timestamp: '1 week ago' },
  { id: '5', title: 'UI Design Feedback', timestamp: '1 week ago' },
  { id: '6', title: 'Performance Issues', timestamp: '2 weeks ago' },
];

// Available AI models
const AI_MODELS = [
  { id: '1', name: 'Llama' },
  { id: '2', name: 'O Llama' },
  { id: '3', name: 'Claude Instant' },
  { id: '4', name: 'DeepSeek Coder' },
  { id: '5', name: 'Gemma 7B' },
  { id: '6', name: 'Mixtral 8x7B' },
];

export default function EnhancedChat() {
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [modelDropdownVisible, setModelDropdownVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState(AI_MODELS[0]);
  const [isNewChat, setIsNewChat] = useState(true);
  const scrollViewRef = useRef(null);
  const windowWidth = Dimensions.get('window').width;
  const isMobile = windowWidth < 768;

  const handleSend = () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setIsNewChat(false);

    // Simulate AI response
    setTimeout(() => {
      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: `This is a response from ${selectedModel.name}. I'm providing an answer based on your question about "${inputMessage.substring(0, 30)}${inputMessage.length > 30 ? '...' : ''}"`,
        isUser: false,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, botMessage]);
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 1000);
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setModelDropdownVisible(false);
  };

  const handleNewChat = () => {
    setMessages([]);
    setIsNewChat(true);
  };

  const getTimeString = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderWelcomeScreen = () => (
    <View style={styles.welcomeContainer}>
      <View style={styles.welcomeContent}>
        <Text style={styles.welcomeTitle}>How can I assist you today?</Text>
        <Text style={styles.welcomeSubtitle}>
          I'm your AI assistant powered by {selectedModel.name}. I can help with:
        </Text>
        
        <View style={styles.capabilitiesContainer}>
          <View style={styles.capabilityItem}>
            <FontAwesome5 name="code" size={24} color="#10a37f" style={styles.capabilityIcon} />
            <View>
              <Text style={styles.capabilityTitle}>Code Assistance</Text>
            </View>
          </View>
          
          <View style={styles.capabilityItem}>
            <FontAwesome5 name="brain" size={24} color="#10a37f" style={styles.capabilityIcon} />
            <View>
              <Text style={styles.capabilityTitle}>Problem Solving</Text>
            </View>
          </View>
          
          <View style={styles.capabilityItem}>
            <FontAwesome5 name="lightbulb" size={24} color="#10a37f" style={styles.capabilityIcon} />
            <View>
              <Text style={styles.capabilityTitle}>Creative Ideas</Text>
            </View>
          </View>
          
          <View style={styles.capabilityItem}>
            <FontAwesome5 name="book" size={24} color="#10a37f" style={styles.capabilityIcon} />
            <View>
              <Text style={styles.capabilityTitle}>Knowledge</Text>
            </View>
          </View>
        </View>
        
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        {/* Header */}
        <View style={styles.header}>
          {isMobile && (
            <TouchableOpacity 
              style={styles.menuButton} 
              onPress={() => setSidebarVisible(!sidebarVisible)}
            >
              <MaterialIcons name="menu" size={24} color="#fff" />
            </TouchableOpacity>
          )}
          
          <TouchableOpacity style={styles.newChatButton} onPress={handleNewChat}>
            <MaterialIcons name="add" size={20} color="#fff" />
            <Text style={styles.newChatButtonText}>New Chat</Text>
          </TouchableOpacity>
          
          <View style={styles.modelSelector}>
            <TouchableOpacity 
              style={styles.modelSelectorButton} 
              onPress={() => setModelDropdownVisible(!modelDropdownVisible)}
            >
              <Text style={styles.modelSelectorText}>{selectedModel.name}</Text>
              <MaterialIcons 
                name={modelDropdownVisible ? "keyboard-arrow-up" : "keyboard-arrow-down"} 
                size={24} 
                color="#fff" 
              />
            </TouchableOpacity>
            
            {modelDropdownVisible && (
              <View style={styles.modelDropdown}>
                {AI_MODELS.map(model => (
                  <TouchableOpacity
                    key={model.id}
                    style={[
                      styles.modelDropdownItem,
                      selectedModel.id === model.id && styles.selectedModelItem
                    ]}
                    onPress={() => handleModelSelect(model)}
                  >
                    <Text 
                      style={[
                        styles.modelDropdownItemText,
                        selectedModel.id === model.id && styles.selectedModelItemText
                      ]}
                    >
                      {model.name}
                    </Text>
                    {selectedModel.id === model.id && (
                      <MaterialIcons name="check" size={20} color="#10a37f" />
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>
        </View>

        {/* Main Content */}
        <View style={styles.contentContainer}>
          {/* Sidebar for chat history (always visible on larger screens) */}
          {(!isMobile || sidebarVisible) && (
            <View style={[
              styles.sidebar,
              isMobile && styles.mobileSidebar,
            ]}>
              <View style={styles.sidebarHeader}>
                <Text style={styles.sidebarTitle}>Chat History</Text>
                {isMobile && (
                  <TouchableOpacity onPress={() => setSidebarVisible(false)}>
                    <MaterialIcons name="close" size={24} color="#fff" />
                  </TouchableOpacity>
                )}
              </View>
              
              <FlatList
                data={CHAT_HISTORY}
                keyExtractor={item => item.id}
                renderItem={({ item }) => (
                  <TouchableOpacity style={styles.historyItem}>
                    <View style={styles.historyItemContent}>
                      <MaterialIcons name="chat" size={20} color="#666" />
                      <View style={styles.historyItemTextContainer}>
                        <Text style={styles.historyItemTitle}>{item.title}</Text>
                        <Text style={styles.historyItemTimestamp}>{item.timestamp}</Text>
                      </View>
                    </View>
                  </TouchableOpacity>
                )}
              />
            </View>
          )}

          {/* Chat area */}
          <View style={styles.chatArea}>
            {isNewChat ? (
              renderWelcomeScreen()
            ) : (
              <ScrollView
                ref={scrollViewRef}
                style={styles.messagesContainer}
                contentContainerStyle={styles.messagesList}
                onContentSizeChange={() => {
                  scrollViewRef.current?.scrollToEnd({ animated: true });
                }}
              >
                {messages.map((message) => (
                  <View
                    key={message.id}
                    style={[
                      styles.messageWrapper,
                      message.isUser ? styles.userMessageWrapper : styles.botMessageWrapper,
                    ]}
                  >
                    <View
                      style={[
                        styles.message,
                        message.isUser ? styles.userMessage : styles.botMessage,
                      ]}
                    >
                      <Text style={styles.messageText}>{message.text}</Text>
                      <Text style={styles.messageTimestamp}>
                        {getTimeString(message.timestamp)}
                      </Text>
                    </View>
                  </View>
                ))}
              </ScrollView>
            )}

            {/* Input area */}
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                value={inputMessage}
                onChangeText={setInputMessage}
                placeholder="Send a message..."
                placeholderTextColor="#666"
                multiline
              />
              <TouchableOpacity 
                style={[
                  styles.sendButton,
                  !inputMessage.trim() && styles.disabledSendButton
                ]} 
                onPress={handleSend}
                disabled={!inputMessage.trim()}
              >
                <MaterialIcons name="send" size={20} color="#fff" />
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
    backgroundColor: '#111',
  },
  menuButton: {
    marginRight: 15,
  },
  newChatButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#333',
    marginRight: 'auto',
  },
  newChatButtonText: {
    color: '#fff',
    marginLeft: 5,
  },
  modelSelector: {
    position: 'relative',
  },
  modelSelectorButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#333',
  },
  modelSelectorText: {
    color: '#fff',
    marginRight: 5,
  },
  modelDropdown: {
    position: 'absolute',
    top: 45,
    right: 0,
    width: 200,
    backgroundColor: '#222',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    zIndex: 10,
    elevation: 5,
  },
  modelDropdownItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  selectedModelItem: {
    backgroundColor: '#1a1a1a',
  },
  modelDropdownItemText: {
    color: '#fff',
  },
  selectedModelItemText: {
    color: '#10a37f',
  },
  contentContainer: {
    flex: 1,
    flexDirection: 'row',
  },
  sidebar: {
    width: 280,
    backgroundColor: '#111',
    borderRightWidth: 1,
    borderRightColor: '#333',
  },
  mobileSidebar: {
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    zIndex: 100,
  },
  sidebarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  sidebarTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  historyItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  historyItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  historyItemTextContainer: {
    marginLeft: 10,
    flex: 1,
  },
  historyItemTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  historyItemTimestamp: {
    color: '#666',
    fontSize: 12,
    marginTop: 3,
  },
  chatArea: {
    flex: 1,
    backgroundColor: '#000',
  },
  welcomeContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  welcomeContent: {
    maxWidth: 800,
    width: '100%',
  },
  welcomeTitle: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 15,
  },
  welcomeSubtitle: {
    color: '#ccc',
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 30,
  },
  capabilitiesContainer: {
    marginBottom: 30,
  },
  capabilityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    backgroundColor: '#111',
    padding: 15,
    borderRadius: 10,
  },
  capabilityIcon: {
    marginRight: 15,
  },
  capabilityTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 5,
  },
  capabilityDescription: {
    color: '#ccc',
    fontSize: 12,
  },
  examplesTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    textAlign: 'center',
  },
  examplesContainer: {
    marginBottom: 20,
  },
  exampleButton: {
    backgroundColor: '#1a1a1a',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#333',
  },
  exampleButtonText: {
    color: '#10a37f',
    fontSize: 16,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesList: {
    padding: 15,
    paddingBottom: 20,
  },
  messageWrapper: {
    marginBottom: 15,
  },
  userMessageWrapper: {
    alignItems: 'flex-end',
  },
  botMessageWrapper: {
    alignItems: 'flex-start',
  },
  message: {
    maxWidth: '80%',
    padding: 15,
    borderRadius: 20,
    position: 'relative',
  },
  userMessage: {
    backgroundColor: '#10a37f',
    borderTopRightRadius: 4,
  },
  botMessage: {
    backgroundColor: '#1a1a1a',
    borderTopLeftRadius: 4,
  },
  messageText: {
    color: '#fff',
    fontSize: 16,
  },
  messageTimestamp: {
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: 12,
    marginTop: 5,
    alignSelf: 'flex-end',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#333',
    backgroundColor: '#111',
  },
  input: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 20,
    padding: 12,
    color: '#fff',
    marginRight: 10,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#10a37f',
    borderRadius: 20,
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabledSendButton: {
    backgroundColor: '#333',
  },
  sendButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
});