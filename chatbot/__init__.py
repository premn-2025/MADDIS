"""
Drug Discovery Chatbot Package

Production-ready Gemini-powered chatbot for interpreting drug discovery results
from the 6 major workflows:
1. RL Molecule Generation
2. Multi-Target RL Optimization
3. Multi-Agent Platform Coordination
4. Property Prediction (ADMET)
5. Real Molecular Docking
6. Chemical Space Analytics
"""

from .chatbot_core import DrugDiscoveryChatbot
from .context_manager import ResultContextManager
from .gemini_client import GeminiClient
from .streamlit_ui import render_chatbot_ui

__all__ = [
    'DrugDiscoveryChatbot',
    'ResultContextManager', 
    'GeminiClient',
    'render_chatbot_ui'
]

__version__ = '1.0.0'
