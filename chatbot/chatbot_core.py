"""
Drug Discovery Chatbot Core

Main orchestrator class that coordinates all components:
- Gemini client for LLM interaction
- Context manager for workflow results
- Interpreters for specialized result formatting
- Intent detection and routing
"""

import logging
from typing import Dict, Any, List, Optional

from .gemini_client import GeminiClient
from .context_manager import ResultContextManager, WorkflowType
from .prompts import get_system_prompt
from .interpreters import (
    RLGenerationInterpreter,
    MultiTargetInterpreter,
    MultiAgentInterpreter,
    PropertyInterpreter,
    DockingInterpreter,
    ChemicalSpaceInterpreter
)

logger = logging.getLogger(__name__)


class DrugDiscoveryChatbot:
    """
    Production-ready chatbot for drug discovery result interpretation
    
    Features:
    - Context-aware responses based on workflow results
    - Specialized interpreters for each workflow type
    - Automatic intent detection and routing
    - Conversation history management
    - Fallback responses when API unavailable
    """
    
    def __init__(self):
        """Initialize the chatbot with all components"""
        self.client = GeminiClient()
        self.context_manager = ResultContextManager()
        
        # Initialize interpreters
        self.interpreters = {
            WorkflowType.RL_GENERATION: RLGenerationInterpreter(),
            WorkflowType.MULTI_TARGET_RL: MultiTargetInterpreter(),
            WorkflowType.MULTI_AGENT: MultiAgentInterpreter(),
            WorkflowType.PROPERTY_PREDICTION: PropertyInterpreter(),
            WorkflowType.DOCKING: DockingInterpreter(),
            WorkflowType.CHEMICAL_SPACE: ChemicalSpaceInterpreter()
        }
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response
        
        Args:
            user_message: The user's input message
            
        Returns:
            The chatbot's response
        """
        # Detect intent and get appropriate interpreter
        workflow_type = self._detect_workflow_intent(user_message)
        
        # Only use workflow-specific prompt/context if we actually have results
        # Otherwise treat it as a general knowledge question
        has_results = (
            workflow_type is not None
            and workflow_type in self.context_manager.results
        )
        
        if has_results:
            system_prompt = get_system_prompt(workflow_type.value)
            context = self.context_manager.get_workflow_context(workflow_type)
        else:
            # General knowledge mode — no workflow grounding
            system_prompt = get_system_prompt(None)
            context = self.context_manager.get_full_context()
        
        # Generate response
        response = self.client.generate_response(
            user_message=user_message,
            system_prompt=system_prompt,
            context=context,
            include_history=True
        )
        
        return response
    
    def _detect_workflow_intent(self, message: str) -> Optional[WorkflowType]:
        """Detect which workflow the user is asking about"""
        message_lower = message.lower()
        
        for wt, interpreter in self.interpreters.items():
            if interpreter.matches_query(message_lower):
                return wt
        
        return None
    
    # ==================== Result Registration ====================
    
    def add_rl_results(
        self,
        molecules: List[Dict],
        training_stats: Dict[str, Any],
        target: str = ""
    ) -> None:
        """Register RL generation results for context"""
        self.context_manager.add_rl_generation_result(
            generated_molecules=molecules,
            training_stats=training_stats,
            target_protein=target
        )
    
    def add_multi_target_results(
        self,
        pareto_solutions: List[Dict],
        targets: List[str],
        history: Dict = None
    ) -> None:
        """Register multi-target optimization results"""
        self.context_manager.add_multi_target_result(
            pareto_solutions=pareto_solutions,
            target_names=targets,
            training_history=history or {}
        )
    
    def add_multiagent_results(
        self,
        agents: Dict[str, Any],
        task_summary: Dict = None,
        recommendations: List[str] = None
    ) -> None:
        """Register multi-agent coordination results"""
        self.context_manager.add_multiagent_result(
            agent_results=agents,
            task_summary=task_summary or {},
            recommendations=recommendations or []
        )
    
    def add_property_results(
        self,
        smiles: str,
        predictions: Dict[str, Any],
        warnings: List[str] = None
    ) -> None:
        """Register property prediction results"""
        self.context_manager.add_property_prediction_result(
            smiles=smiles,
            predictions=predictions,
            warnings=warnings
        )
    
    def add_docking_results(
        self,
        smiles: str,
        target: str,
        affinity: float,
        interactions: Dict = None,
        confidence: str = ""
    ) -> None:
        """Register molecular docking results"""
        self.context_manager.add_docking_result(
            smiles=smiles,
            target_protein=target,
            binding_affinity=affinity,
            interactions=interactions or {},
            confidence=confidence
        )
    
    def add_chemical_space_results(
        self,
        molecules_count: int,
        clusters: List[Dict],
        diversity_metrics: Dict[str, float]
    ) -> None:
        """Register chemical space analysis results"""
        self.context_manager.add_chemical_space_result(
            molecules_analyzed=molecules_count,
            clusters=clusters,
            diversity_metrics=diversity_metrics
        )
    
    # ==================== Utility Methods ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get chatbot status information"""
        return {
            "api_available": self.client.is_available,
            "model": self.client.config.gemini_model,
            "conversation_length": len(self.client.conversation_history),
            "results_stored": self.context_manager.get_summary()
        }
    
    def check_api_status(self) -> str:
        """Check and return API connection status"""
        status = self.get_status()
        if status["api_available"]:
            return f"✅ Gemini API connected ({status['model']})"
        return "❌ Gemini API not available - using fallback mode"
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.client.clear_history()
    
    def clear_results(self, workflow_type: str = None) -> None:
        """Clear stored results"""
        if workflow_type:
            wt = WorkflowType(workflow_type)
            self.context_manager.clear_results(wt)
        else:
            self.context_manager.clear_results()
    
    def get_quick_questions(self) -> List[str]:
        """Get suggested questions based on stored results"""
        questions = []
        summary = self.context_manager.get_summary()
        
        if summary.get('rl_generation', 0) > 0:
            questions.append("What is the best molecule generated?")
            questions.append("How did the RL training progress?")
        
        if summary.get('multi_target_rl', 0) > 0:
            questions.append("Explain the Pareto-optimal solutions")
            questions.append("Which molecule has the best multi-target profile?")
        
        if summary.get('property_prediction', 0) > 0:
            questions.append("Is this molecule drug-like?")
            questions.append("What are the ADMET concerns?")
        
        if summary.get('docking', 0) > 0:
            questions.append("Explain the binding affinity results")
            questions.append("What interactions does the molecule make?")
        
        if summary.get('chemical_space', 0) > 0:
            questions.append("How diverse are the generated molecules?")
            questions.append("What do the clusters represent?")
        
        # Default questions if no results
        if not questions:
            questions = [
                "What are ADMET properties?",
                "Explain binding affinity",
                "What is drug-likeness?"
            ]
        
        return questions[:5]
