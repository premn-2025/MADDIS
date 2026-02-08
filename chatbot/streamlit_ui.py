"""Streamlit UI Component for Drug Discovery Chatbot"""

import streamlit as st
from typing import Optional, Callable

from .chatbot_core import DrugDiscoveryChatbot


def render_chatbot_ui(
    chatbot: Optional[DrugDiscoveryChatbot] = None,
    container_height: int = 500,
    show_status: bool = True,
    show_quick_questions: bool = True,
    on_message_callback: Optional[Callable] = None
) -> None:
    """
    Render the chatbot UI component in Streamlit.
    Uses standard Streamlit chat pattern for reliable rendering.
    """
    # Initialize or get existing chatbot
    if 'drug_chatbot' not in st.session_state:
        st.session_state.drug_chatbot = chatbot or DrugDiscoveryChatbot()
    chatbot = st.session_state.drug_chatbot

    # Initialize chat history
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []

    # Pending query from quick-question buttons (set on click, consumed on next rerun)
    if 'pending_chat_query' not in st.session_state:
        st.session_state.pending_chat_query = None

    # ── Header with status ──
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🤖 Drug Discovery AI Assistant")
    with col2:
        if show_status:
            status = chatbot.get_status()
            if status["api_available"]:
                st.success("●  Online", icon="✅")
            else:
                st.warning("●  Offline", icon="⚠️")
                with st.expander("ℹ️ Why offline?", expanded=False):
                    if not chatbot.client._initialization_attempted:
                        chatbot.client._initialize_client()
                    if chatbot.client.is_available:
                        st.info("✅ Chatbot is now online! Refresh to see updated status.")
                    else:
                        st.info("""
**Fallback Mode Active** — works without API for basic questions.

**To enable full AI mode:**
1. Get API key: https://aistudio.google.com/app/apikey
2. Add to `.env`: `GEMINI_API_KEY=your_key_here`
3. Refresh the page
""")

    # ── Quick questions ──
    if show_quick_questions:
        questions = chatbot.get_quick_questions()
        if questions:
            st.caption("💡 Suggested questions:")
            cols = st.columns(min(len(questions), 3))
            for i, q in enumerate(questions[:3]):
                with cols[i]:
                    if st.button(q[:35] + "..." if len(q) > 35 else q, key=f"quick_q_{i}", use_container_width=True):
                        st.session_state.pending_chat_query = q

    # ── Determine the active query for this run ──
    # Quick-question click (consumed once)
    active_query = st.session_state.pending_chat_query
    if active_query:
        st.session_state.pending_chat_query = None

    # ── Chat container ──
    try:
        chat_container = st.container(height=container_height)
    except TypeError:
        chat_container = st.container()

    with chat_container:
        # Display persisted history
        for msg in st.session_state.chatbot_messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

        # Welcome message when empty
        if not st.session_state.chatbot_messages and not active_query:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
👋 **Welcome!** I'm your AI assistant for drug discovery analysis.

I can help you:
- 🧬 Interpret molecular generation results
- 💊 Explain ADMET and drug-likeness predictions
- 🔗 Analyze binding affinity and docking results
- 📊 Understand chemical space analyses

Ask me anything about your drug discovery results!
""")

        # Process quick-question if one was clicked
        if active_query:
            _handle_and_render(chatbot, active_query, chat_container, on_message_callback)

    # ── Chat input ──
    user_input = st.chat_input("Ask about your drug discovery results...", key="chatbot_input")
    if user_input:
        _handle_and_render(chatbot, user_input, chat_container, on_message_callback)

    # ── Clear button ──
    _, _, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chatbot_messages = []
            chatbot.clear_history()
            st.rerun()


def _handle_and_render(
    chatbot: DrugDiscoveryChatbot,
    user_input: str,
    container,
    callback: Optional[Callable] = None,
) -> None:
    """Standard Streamlit chat pattern: render user msg, call API with spinner, render response."""
    with container:
        # Show user bubble
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Show assistant bubble with spinner while API runs
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔬 Analyzing..."):
                try:
                    response = chatbot.chat(user_input)
                except Exception as e:
                    response = f"❌ Error: {e}"
            st.markdown(response)

    # Persist so the history loop shows them on next rerun
    st.session_state.chatbot_messages.append({"role": "user", "content": user_input})
    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

    if callback:
        callback(user_input, response)


def render_chatbot_sidebar(chatbot: Optional[DrugDiscoveryChatbot] = None) -> None:
    """
    Render a compact chatbot in the sidebar
    
    Args:
        chatbot: Optional existing chatbot instance
    """
    with st.sidebar:
        st.subheader("💬 AI Assistant")
        
        # Initialize chatbot
        if 'sidebar_chatbot' not in st.session_state:
            st.session_state.sidebar_chatbot = chatbot or DrugDiscoveryChatbot()
        
        chatbot = st.session_state.sidebar_chatbot
        
        # Status
        status = chatbot.get_status()
        st.caption(f"{'🟢' if status['api_available'] else '🔴'} {status['model']}")
        
        # Simple input
        user_input = st.text_input("Ask a question:", key="sidebar_chat_input")
        
        if user_input:
            with st.spinner("Thinking..."):
                response = chatbot.chat(user_input)
            st.info(response)


def create_chatbot_expander(
    chatbot: Optional[DrugDiscoveryChatbot] = None,
    expanded: bool = False
) -> None:
    """
    Create a chatbot in an expander widget
    
    Args:
        chatbot: Optional existing chatbot instance
        expanded: Whether the expander is initially expanded
    """
    with st.expander("🤖 AI Assistant - Ask about your results", expanded=expanded):
        # Initialize
        if 'expander_chatbot' not in st.session_state:
            st.session_state.expander_chatbot = chatbot or DrugDiscoveryChatbot()
        
        chatbot = st.session_state.expander_chatbot
        
        if 'expander_messages' not in st.session_state:
            st.session_state.expander_messages = []
        
        # Display last 5 messages
        for msg in st.session_state.expander_messages[-5:]:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")
        
        # Input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Your question:", key="expander_input", label_visibility="collapsed")
        with col2:
            send_btn = st.button("Send", type="primary", use_container_width=True)
        
        if send_btn and user_input:
            st.session_state.expander_messages.append({"role": "user", "content": user_input})
            response = chatbot.chat(user_input)
            st.session_state.expander_messages.append({"role": "assistant", "content": response})
            st.rerun()
