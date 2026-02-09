"""
Configuration Management for Drug Discovery Chatbot

Optimized settings for accuracy and speed.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Load environment variables
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Load Streamlit secrets into environment variables (for deployed apps)
try:
    import streamlit as st
    for key, value in st.secrets.items():
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = str(value)
except Exception:
    pass


@dataclass
class ChatbotConfig:
    """Optimized configuration settings"""
    
    # API Settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    
    # Local Model Settings (Ollama)
    model_provider: str = "gemini"  # "gemini" | "ollama" | "hybrid"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"
    
    # Optimized for accuracy (low temperature)
    temperature: float = 0.1
    max_output_tokens: int = 1024
    top_p: float = 0.8
    top_k: int = 20
    
    # Conversation limits
    max_history_length: int = 6  # 3 exchanges
    context_max_chars: int = 1200
    
    @classmethod
    def from_env(cls) -> 'ChatbotConfig':
        """Create from environment"""
        return cls(
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview'),
            model_provider=os.getenv('MODEL_PROVIDER', 'gemini'),
            ollama_host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
            ollama_model=os.getenv('OLLAMA_MODEL', 'gemma2:9b'),
            temperature=float(os.getenv('CHATBOT_TEMPERATURE', '0.1')),
            max_output_tokens=int(os.getenv('CHATBOT_MAX_TOKENS', '1024'))
        )
    
    def validate(self) -> tuple[bool, str]:
        """Validate configuration"""
        if not self.gemini_api_key:
            return False, "GEMINI_API_KEY not found"
        if not self.gemini_api_key.startswith('AIza'):
            return False, "Invalid GEMINI_API_KEY format"
        return True, "Valid"


_config: Optional[ChatbotConfig] = None


def get_config() -> ChatbotConfig:
    """Get global configuration"""
    global _config
    if _config is None:
        _config = ChatbotConfig.from_env()
    return _config


def reload_config() -> ChatbotConfig:
    """Reload configuration"""
    global _config
    _config = ChatbotConfig.from_env()
    return _config
