"""
Chat History Manager for Copilot-Style Chatbot
Handles persistent storage of chat sessions
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ChatMessage:
    """A single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChatSession:
    """A chat session with messages"""
    id: str
    title: str
    created_at: str
    messages: List[Dict]
    
    @classmethod
    def create_new(cls, first_message: str = "") -> 'ChatSession':
        """Create a new chat session"""
        title = first_message[:40] + "..." if len(first_message) > 40 else first_message
        if not title:
            title = f"Chat {datetime.now().strftime('%H:%M')}"
        return cls(
            id=str(uuid.uuid4())[:8],
            title=title,
            created_at=datetime.now().isoformat(),
            messages=[]
        )


class ChatHistoryManager:
    """Manages chat history persistence"""
    
    def __init__(self, history_dir: str = None):
        """Initialize with optional custom history directory"""
        if history_dir is None:
            # Default to chatbot/history folder
            base_dir = os.path.dirname(os.path.abspath(__file__))
            history_dir = os.path.join(base_dir, "history")
        
        self.history_dir = history_dir
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self):
        """Create history directory if it doesn't exist"""
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
    
    def _session_path(self, session_id: str) -> str:
        """Get path to session file"""
        return os.path.join(self.history_dir, f"{session_id}.json")
    
    def save_session(self, session: ChatSession) -> bool:
        """Save a chat session to file"""
        try:
            path = self._session_path(session.id)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a chat session from file"""
        try:
            path = self._session_path(session_id)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ChatSession(**data)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            # Corrupted session file — delete it so it stops causing errors
            print(f"Corrupted session {session_id}, removing: {e}")
            try:
                os.remove(self._session_path(session_id))
            except OSError:
                pass
            return None
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """List all chat sessions, newest first"""
        sessions = []
        try:
            for filename in os.listdir(self.history_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]
                    session = self.load_session(session_id)
                    if session:
                        sessions.append({
                            "id": session.id,
                            "title": session.title,
                            "created_at": session.created_at,
                            "message_count": len(session.messages)
                        })
            
            # Sort by created_at descending
            sessions.sort(key=lambda x: x["created_at"], reverse=True)
            return sessions[:limit]
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        try:
            path = self._session_path(session_id)
            if os.path.exists(path):
                os.remove(path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Delete all chat sessions"""
        try:
            for filename in os.listdir(self.history_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.history_dir, filename))
            return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
