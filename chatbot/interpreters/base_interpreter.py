"""
Base Interpreter for Drug Discovery Results

Abstract base class that all workflow interpreters inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseInterpreter(ABC):
    """
    Abstract base class for workflow result interpreters
    """
    
    @property
    @abstractmethod
    def workflow_type(self) -> str:
        """Return the workflow type this interpreter handles"""
        pass
    
    @property
    @abstractmethod
    def keywords(self) -> List[str]:
        """Keywords that trigger this interpreter"""
        pass
    
    @abstractmethod
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format result data into LLM-friendly context string"""
        pass
    
    @abstractmethod
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        pass
    
    def matches_query(self, query: str) -> bool:
        """Check if a query matches this interpreter's domain"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.keywords)
    
    def format_metric(self, value: Any, precision: int = 2) -> str:
        """Format a metric value for display"""
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)
