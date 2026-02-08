"""
Groq API Provider - Ultra-fast inference

Groq provides extremely fast LLM inference using their LPU hardware.
Much faster than Gemini or local models.
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq API"""
    api_key: str = ""
    model: str = "llama-3.1-70b-versatile"  # Fast and capable
    temperature: float = 0.1
    max_tokens: int = 1024
    
    @classmethod
    def from_env(cls) -> 'GroqConfig':
        """Load from environment variables"""
        return cls(
            api_key=os.getenv('GROQ_API_KEY', ''),
            model=os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile'),
            temperature=float(os.getenv('GROQ_TEMPERATURE', '0.1')),
            max_tokens=int(os.getenv('GROQ_MAX_TOKENS', '1024'))
        )


class GroqClient:
    """
    Client for Groq API - Ultra-fast LLM inference
    
    Much faster than Gemini or local models due to Groq's LPU hardware.
    """
    
    def __init__(self, config: Optional[GroqConfig] = None):
        self.config = config or GroqConfig.from_env()
        self.client = None
        self.is_available = False
        self._init_client()
    
    def _init_client(self):
        """Initialize Groq client"""
        if not self.config.api_key:
            logger.warning("GROQ_API_KEY not set")
            self.is_available = False
            return
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.config.api_key)
            self.is_available = True
            logger.info(f"✅ Groq client initialized with model: {self.config.model}")
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            self.is_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.is_available = False
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        include_history: bool = False
    ) -> str:
        """
        Generate a response using Groq API
        """
        if not self.is_available or not self.client:
            return "⚠️ Groq API not available. Check GROQ_API_KEY."
        
        # Build messages
        messages = []
        
        # System message
        if system_prompt or context:
            system_content = system_prompt or "You are a helpful drug discovery assistant."
            if context:
                system_content += f"\n\nCONTEXT:\n{context[:2000]}"
            messages.append({"role": "system", "content": system_content})
        
        # User message
        messages.append({"role": "user", "content": user_message})
        
        try:
            logger.info(f"🚀 Calling Groq API ({self.config.model})...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=10.0  # 10 second timeout for "instant" model
            )
            
            elapsed = time.time() - start_time
            response_text = response.choices[0].message.content.strip()
            
            logger.info(f"✅ Groq response in {elapsed:.2f}s ({len(response_text)} chars)")
            return response_text
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"⚠️ Groq API error: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "provider": "groq",
            "is_available": self.is_available,
            "model": self.config.model
        }


# Quick test
if __name__ == "__main__":
    print("Testing Groq API...")
    client = GroqClient()
    print(f"Status: {client.get_status()}")
    
    if client.is_available:
        response = client.generate_response(
            "What is Lipinski's Rule of 5?",
            system_prompt="You are a drug discovery expert."
        )
        print(f"\nResponse:\n{response}")
