"""
GitHub Copilot-Style Chat UI for Drug Discovery Platform
A modern, sleek chat interface with right-side panel and history
"""

import streamlit as st
from typing import Optional
from datetime import datetime

from .chat_history import ChatHistoryManager, ChatSession, ChatMessage
from .chatbot_core import DrugDiscoveryChatbot


# Custom CSS for Copilot-like styling
COPILOT_CSS = """
<style>
/* Chat panel container */
.copilot-panel {
    background: linear-gradient(180deg, #1e1e2e 0%, #181825 100%);
    border-left: 1px solid #313244;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Header styling */
.copilot-header {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.copilot-header h3 {
    margin: 0;
    color: white;
    font-size: 16px;
    font-weight: 600;
}

/* Message bubbles */
.copilot-message {
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 90%;
    word-wrap: break-word;
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.copilot-user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.copilot-assistant {
    background: #2a2a3c;
    color: #e2e8f0;
    border: 1px solid #3f3f5a;
    border-bottom-left-radius: 4px;
}

/* History item styling */
.history-item {
    background: #252536;
    border: 1px solid #3f3f5a;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.history-item:hover {
    background: #2f2f45;
    border-color: #6366f1;
}

.history-title {
    color: #e2e8f0;
    font-weight: 500;
    font-size: 14px;
    margin-bottom: 4px;
}

.history-meta {
    color: #71717a;
    font-size: 12px;
}

/* Chat input area */
.chat-input-container {
    background: #1e1e2e;
    border-top: 1px solid #313244;
    padding: 12px;
    position: sticky;
    bottom: 0;
}

/* Status indicator */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
}

.status-offline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
}

/* New chat button */
.new-chat-btn {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.new-chat-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

/* Scrollable areas */
.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    max-height: 400px;
}

/* Avatar styling */
.avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

.avatar-user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}

.avatar-bot {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
}
</style>
"""


def _init_session_state():
    """Initialize session state for chat"""
    if 'copilot_open' not in st.session_state:
        st.session_state.copilot_open = False
    
    if 'copilot_chatbot' not in st.session_state:
        st.session_state.copilot_chatbot = DrugDiscoveryChatbot()
    
    if 'copilot_history_mgr' not in st.session_state:
        st.session_state.copilot_history_mgr = ChatHistoryManager()
    
    if 'copilot_current_session' not in st.session_state:
        st.session_state.copilot_current_session = None
    
    if 'copilot_messages' not in st.session_state:
        st.session_state.copilot_messages = []
    
    if 'copilot_view' not in st.session_state:
        st.session_state.copilot_view = 'chat'  # 'chat' or 'history'
    
    if 'copilot_last_input' not in st.session_state:
        st.session_state.copilot_last_input = ""


def _toggle_panel():
    """Toggle chat panel open/closed"""
    st.session_state.copilot_open = not st.session_state.copilot_open


def _new_chat():
    """Start a new chat session"""
    # Save current session if it has messages
    if st.session_state.copilot_messages and st.session_state.copilot_current_session:
        st.session_state.copilot_current_session.messages = st.session_state.copilot_messages
        st.session_state.copilot_history_mgr.save_session(st.session_state.copilot_current_session)
    
    # Create new session
    st.session_state.copilot_current_session = ChatSession.create_new()
    st.session_state.copilot_messages = []
    st.session_state.copilot_view = 'chat'


def _load_session(session_id: str):
    """Load a session from history"""
    session = st.session_state.copilot_history_mgr.load_session(session_id)
    if session:
        st.session_state.copilot_current_session = session
        st.session_state.copilot_messages = session.messages
        st.session_state.copilot_view = 'chat'


def _send_message(user_input: str):
    """Send a message and get response. No processing flag — safe against Streamlit reruns."""
    if not user_input.strip():
        return
    
    st.session_state.copilot_last_input = user_input
    
    # Ensure we have a session
    if not st.session_state.copilot_current_session:
        st.session_state.copilot_current_session = ChatSession.create_new(user_input)
    
    # Add user message
    st.session_state.copilot_messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get response
    try:
        response = st.session_state.copilot_chatbot.chat(user_input)
    except Exception as e:
        response = f"Error: {str(e)}"
    
    # Add assistant message
    st.session_state.copilot_messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update session title if first message
    if len(st.session_state.copilot_messages) == 2:
        title = user_input[:40] + "..." if len(user_input) > 40 else user_input
        st.session_state.copilot_current_session.title = title
    
    # Auto-save
    st.session_state.copilot_current_session.messages = st.session_state.copilot_messages
    st.session_state.copilot_history_mgr.save_session(st.session_state.copilot_current_session)


def render_copilot_chat():
    """
    Render the GitHub Copilot-style chat interface
    Call this at the end of your main Streamlit app
    """
    _init_session_state()
    
    # Inject CSS
    st.markdown(COPILOT_CSS, unsafe_allow_html=True)
    
    # Create the right-side chat panel using columns
    # Main content takes most space, chat panel on right when open
    
    if st.session_state.copilot_open:
        # Split into main (empty placeholder) and chat column
        main_col, chat_col = st.columns([3, 1])
        
        with chat_col:
            _render_chat_panel()
    
    # Floating chat button (in sidebar for Streamlit compatibility)
    with st.sidebar:
        st.markdown("---")
        
        # Chat toggle button
        if st.session_state.copilot_open:
            if st.button("✖️ Close AI Chat", key="copilot_close_btn", use_container_width=True):
                st.session_state.copilot_open = False
                st.rerun()
        else:
            if st.button("🤖 AI Assistant", key="copilot_open_btn", use_container_width=True, type="primary"):
                st.session_state.copilot_open = True
                st.rerun()
        
        # Quick status - show actual provider
        chatbot = st.session_state.copilot_chatbot
        status = chatbot.get_status()
        provider = chatbot.client.config.model_provider if hasattr(chatbot, 'client') else 'unknown'
        if provider == 'groq':
            st.caption("🚀 Groq API (Fast)")
        elif provider == 'ollama':
            st.caption("🤖 Local Ollama")
        elif status["api_available"]:
            st.caption("🟢 Gemini API")
        else:
            st.caption("🔴 Offline Mode")


def _render_chat_panel():
    """Render the chat panel content"""
    chatbot = st.session_state.copilot_chatbot
    history_mgr = st.session_state.copilot_history_mgr
    
    # Header
    st.markdown("""
    <div class="copilot-header">
        <span style="font-size: 20px;">🤖</span>
        <h3>Drug Discovery AI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Status - show actual provider
    status = chatbot.get_status()
    provider = chatbot.client.config.model_provider if hasattr(chatbot, 'client') else 'unknown'
    if provider == 'groq':
        st.markdown('<span class="status-online">🚀 Groq Fast</span>', unsafe_allow_html=True)
    elif provider == 'ollama':
        st.markdown('<span class="status-online">🤖 Local</span>', unsafe_allow_html=True)
    elif status["api_available"]:
        st.markdown('<span class="status-online">● Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">● Offline</span>', unsafe_allow_html=True)
    
    # View toggle
    tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])
    
    with tab1:
        _render_chat_view()
    
    with tab2:
        _render_history_view()


def _render_chat_view():
    """Render the chat conversation view"""
    # New chat button
    if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
        _new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Messages container
    messages_container = st.container()
    
    with messages_container:
        if not st.session_state.copilot_messages:
            # Welcome message
            st.markdown("""
            <div class="copilot-message copilot-assistant">
                👋 <strong>Welcome!</strong><br><br>
                I'm your AI assistant for drug discovery. Ask me about:
                <ul>
                    <li>🧬 Molecular analysis results</li>
                    <li>💊 ADMET predictions</li>
                    <li>🔗 Binding affinity</li>
                    <li>📊 RL generation insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display messages
            for msg in st.session_state.copilot_messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="copilot-message copilot-user">
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="copilot-message copilot-assistant">
                        🤖 {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input area - use text_input instead of chat_input (which can't be in tabs)
    st.markdown("---")
    col_input, col_send = st.columns([4, 1])
    with col_input:
        user_input = st.text_input("Your question:", key="copilot_text_input", 
                                   placeholder="Ask about drug discovery...", label_visibility="collapsed")
    with col_send:
        send_clicked = st.button("🚀", key="copilot_send_btn", type="primary", use_container_width=True)
    
    if send_clicked and user_input:
        with st.spinner("🤖 Thinking..."):
            _send_message(user_input)
        st.rerun()


def _render_history_view():
    """Render the chat history view"""
    history_mgr = st.session_state.copilot_history_mgr
    
    # Get history list
    sessions = history_mgr.list_sessions(limit=15)
    
    if not sessions:
        st.info("No chat history yet. Start a conversation!")
        return
    
    st.markdown("### Recent Chats")
    
    for session in sessions:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Format date
            try:
                dt = datetime.fromisoformat(session["created_at"])
                date_str = dt.strftime("%b %d, %H:%M")
            except:
                date_str = "Unknown"
            
            if st.button(
                f"💬 {session['title']}\n\n_{date_str} • {session['message_count']} msgs_",
                key=f"load_{session['id']}",
                use_container_width=True
            ):
                _load_session(session["id"])
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{session['id']}", help="Delete"):
                history_mgr.delete_session(session["id"])
                st.rerun()
    
    # Clear all button
    st.markdown("---")
    if st.button("🗑️ Clear All History", key="clear_all_history", type="secondary"):
        history_mgr.clear_all()
        st.rerun()


# Convenience function for minimal integration
def add_copilot_chat():
    """
    Simple function to add Copilot chat to your app.
    Just call add_copilot_chat() at the end of your main app.
    """
    render_copilot_chat()
