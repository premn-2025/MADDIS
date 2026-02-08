"""
Local Model Provider - Ollama/Gemma Integration

Provides local LLM inference as a fallback when Gemini API is unavailable.
Uses Ollama to run Google's Gemma 2 model locally.
"""

import os
import logging
import time
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class OllamaConfig:
    """Configuration for Ollama local model"""
    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 512  # Shorter for speed
    timeout: int = 45
    
    @classmethod
    def from_env(cls) -> 'OllamaConfig':
        """Load from environment variables"""
        return cls(
            host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
            model=os.getenv('OLLAMA_MODEL', 'gemma2:9b'),
            temperature=float(os.getenv('OLLAMA_TEMPERATURE', '0.1')),
            max_tokens=int(os.getenv('OLLAMA_MAX_TOKENS', '1024')),
            timeout=int(os.getenv('OLLAMA_TIMEOUT', '60'))
        )


class OllamaClient:
    """
    Client for Ollama local LLM inference
    
    Provides the same interface as GeminiClient for easy swapping.
    Uses Gemma 2 (Google's open-source model) by default.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig.from_env()
        self.is_available = False
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.config.host}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning("Ollama server not responding")
                self.is_available = False
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Handle model name variations (gemma2:9b vs gemma2:9b-instruct-q4_0)
            model_base = self.config.model.split(':')[0]
            has_model = any(model_base in name for name in model_names)
            
            if has_model:
                logger.info(f"✅ Ollama available with model: {self.config.model}")
                self.is_available = True
                return True
            else:
                logger.warning(f"Model {self.config.model} not found. Available: {model_names}")
                logger.info(f"💡 Run: ollama pull {self.config.model}")
                self.is_available = False
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not running. Start with: ollama serve")
            self.is_available = False
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            self.is_available = False
            return False
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        include_history: bool = False  # Not supported for simplicity
    ) -> str:
        """
        Generate a response using local Ollama model
        
        Same interface as GeminiClient for easy swapping.
        """
        if not self.is_available:
            self._check_availability()
            if not self.is_available:
                return "⚠️ Local model not available. Please start Ollama with: ollama serve"
        
        # Build prompt
        full_prompt = self._build_prompt(user_message, system_prompt, context)
        
        # Call Ollama API
        try:
            response = self._call_ollama(full_prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"⚠️ Local model error: {str(e)}"
    
    def _build_prompt(self, user_message: str, system_prompt: str, context: str) -> str:
        """Build prompt for Ollama"""
        parts = []
        
        if system_prompt:
            parts.append(f"SYSTEM: {system_prompt}")
        
        if context:
            parts.append(f"CONTEXT:\n{context[:1500]}")  # Limit context size
        
        parts.append(f"USER: {user_message}")
        parts.append("ASSISTANT:")
        
        return "\n\n".join(parts)
    
    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama"""
        url = f"{self.config.host}/api/generate"
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        logger.info(f"🤖 Calling local Ollama ({self.config.model})...")
        start_time = time.time()
        
        response = requests.post(
            url, 
            json=payload, 
            timeout=self.config.timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.status_code} - {response.text}")
        
        result = response.json()
        response_text = result.get('response', '').strip()
        
        logger.info(f"✅ Ollama response in {elapsed:.1f}s ({len(response_text)} chars)")
        
        return response_text
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        self._check_availability()
        return {
            "provider": "ollama",
            "is_available": self.is_available,
            "model": self.config.model,
            "host": self.config.host
        }


class HybridModelProvider:
    """
    Hybrid model provider that tries Gemini API first, falls back to local Ollama
    
    This ensures the chatbot always works:
    1. Try Gemini API (fast, high quality)
    2. If rate limited or error, use local Gemma via Ollama
    """
    
    def __init__(self):
        self.gemini_client = None
        self.ollama_client = None
        self.primary_provider = os.getenv('MODEL_PROVIDER', 'gemini').lower()
        
        self._init_providers()
    
    def _init_providers(self):
        """Initialize both providers"""
        # Try to init Gemini
        try:
            from .gemini_client import GeminiClient
            self.gemini_client = GeminiClient()
            logger.info("Gemini client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini: {e}")
        
        # Init Ollama
        try:
            self.ollama_client = OllamaClient()
            if self.ollama_client.is_available:
                logger.info("Ollama client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Ollama: {e}")
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        include_history: bool = True
    ) -> str:
        """
        Generate response using best available provider
        """
        # Determine which provider to try first
        if self.primary_provider == 'ollama':
            return self._try_ollama_then_gemini(user_message, system_prompt, context, include_history)
        else:
            return self._try_gemini_then_ollama(user_message, system_prompt, context, include_history)
    
    def _try_gemini_then_ollama(
        self, user_message: str, system_prompt: str, context: str, include_history: bool
    ) -> str:
        """Try Gemini first, fall back to Ollama"""
        # Try Gemini
        if self.gemini_client and self.gemini_client.is_available:
            try:
                response = self.gemini_client.generate_response(
                    user_message, system_prompt, context, include_history
                )
                if response and not response.startswith("ℹ️ Please run"):
                    return response
            except Exception as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['rate', 'quota', '429', 'limit']):
                    logger.warning("Gemini rate limited, falling back to Ollama")
                else:
                    logger.error(f"Gemini error: {e}")
        
        # Fall back to Ollama
        if self.ollama_client and self.ollama_client.is_available:
            logger.info("Using Ollama fallback")
            return self.ollama_client.generate_response(
                user_message, system_prompt, context
            )
        
        return "⚠️ No AI providers available. Check Gemini API key or start Ollama."
    
    def _try_ollama_then_gemini(
        self, user_message: str, system_prompt: str, context: str, include_history: bool
    ) -> str:
        """Try Ollama first, fall back to Gemini"""
        # Try Ollama
        if self.ollama_client and self.ollama_client.is_available:
            try:
                response = self.ollama_client.generate_response(
                    user_message, system_prompt, context
                )
                if response and not response.startswith("⚠️"):
                    return response
            except Exception as e:
                logger.warning(f"Ollama error: {e}, trying Gemini")
        
        # Fall back to Gemini
        if self.gemini_client and self.gemini_client.is_available:
            return self.gemini_client.generate_response(
                user_message, system_prompt, context, include_history
            )
        
        return "⚠️ No AI providers available."
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            "primary": self.primary_provider,
            "gemini": self.gemini_client.get_status() if self.gemini_client else {"is_available": False},
            "ollama": self.ollama_client.get_status() if self.ollama_client else {"is_available": False}
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Ollama Local Model Provider")
    print("=" * 60)
    
    client = OllamaClient()
    
    print(f"\nStatus: {client.get_status()}")
    
    if client.is_available:
        print("\n🧪 Testing generation...")
        response = client.generate_response(
            user_message="What is Lipinski's Rule of 5?",
            system_prompt="You are a drug discovery expert.",
            context=""
        )
        print(f"\nResponse:\n{response}")
    else:
        print("\n⚠️ Ollama not available")
        print("   1. Start Ollama: ollama serve")
        print("   2. Pull model: ollama pull gemma2:9b")
