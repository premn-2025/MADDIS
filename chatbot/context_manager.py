"""
Optimized Context Manager for Drug Discovery Chatbot

Efficient result storage with token-optimized context generation.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class WorkflowType(Enum):
    """Types of drug discovery workflows"""
    RL_GENERATION = "rl_generation"
    MULTI_TARGET_RL = "multi_target_rl"
    MULTI_AGENT = "multi_agent"
    PROPERTY_PREDICTION = "property_prediction"
    DOCKING = "docking"
    CHEMICAL_SPACE = "chemical_space"


@dataclass
class WorkflowResult:
    """Container for workflow results"""
    workflow_type: WorkflowType
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class ResultContextManager:
    """
    Optimized context manager with efficient token usage
    """
    
    def __init__(self):
        """Initialize context manager"""
        self.results: Dict[WorkflowType, WorkflowResult] = {}  # Only keep latest per type
        self.current_molecule: Optional[str] = None
        self.current_target: Optional[str] = None
    
    # ==================== Optimized Result Storage ====================
    
    def add_rl_generation_result(
        self,
        generated_molecules: List[Dict[str, Any]],
        training_stats: Dict[str, Any],
        target_protein: str = ""
    ) -> None:
        """Store RL generation results (only essential data)"""
        best = max(generated_molecules, key=lambda x: x.get('total_reward', 0)) if generated_molecules else {}
        
        self.results[WorkflowType.RL_GENERATION] = WorkflowResult(
            workflow_type=WorkflowType.RL_GENERATION,
            data={
                "count": len(generated_molecules),
                "target": target_protein,
                "best_smiles": best.get('smiles', '')[:60],
                "best_reward": round(best.get('total_reward', 0), 3),
                "binding": round(best.get('rewards', {}).get('binding_affinity', 0), 3),
                "drug_likeness": round(best.get('rewards', {}).get('drug_likeness', 0), 3),
                "valid_rate": round(training_stats.get('valid_rate', 0), 2)
            },
            summary=f"Generated {len(generated_molecules)} molecules for {target_protein}"
        )
    
    def add_multi_target_result(
        self,
        pareto_solutions: List[Dict[str, Any]],
        target_names: List[str],
        training_history: Dict[str, Any]
    ) -> None:
        """Store multi-target results"""
        best = pareto_solutions[0] if pareto_solutions else {}
        
        self.results[WorkflowType.MULTI_TARGET_RL] = WorkflowResult(
            workflow_type=WorkflowType.MULTI_TARGET_RL,
            data={
                "targets": target_names,
                "pareto_count": len(pareto_solutions),
                "best_smiles": best.get('smiles', '')[:60],
                "affinities": {k: round(v, 2) for k, v in best.get('target_affinities', {}).items()},
                "qed": round(best.get('qed_score', 0), 3),
                "reward": round(best.get('total_reward', 0), 3)
            },
            summary=f"{len(pareto_solutions)} Pareto solutions for {', '.join(target_names)}"
        )
    
    def add_multiagent_result(
        self,
        agent_results: Dict[str, Any],
        task_summary: Dict[str, Any],
        recommendations: List[str]
    ) -> None:
        """Store multi-agent results"""
        self.results[WorkflowType.MULTI_AGENT] = WorkflowResult(
            workflow_type=WorkflowType.MULTI_AGENT,
            data={
                "agents": {k: {"status": v.get("status"), "score": round(v.get("score", 0), 2)} 
                          for k, v in agent_results.items()},
                "time_sec": round(task_summary.get('total_execution_time_seconds', 0), 1),
                "recommendations": recommendations[:3]
            },
            summary=f"{len(agent_results)} agents completed analysis"
        )
    
    def add_property_prediction_result(
        self,
        smiles: str,
        predictions: Dict[str, Any],
        warnings: List[str] = None
    ) -> None:
        """Store property prediction results"""
        self.current_molecule = smiles[:50]
        
        self.results[WorkflowType.PROPERTY_PREDICTION] = WorkflowResult(
            workflow_type=WorkflowType.PROPERTY_PREDICTION,
            data={
                "smiles": smiles[:50],
                "drug_like": predictions.get('is_drug_like', False),
                "qed": round(predictions.get('qed_score', 0), 3),
                "toxicity": round(predictions.get('toxicity_score', 0), 3),
                "lipinski": predictions.get('lipinski_violations', 0),
                "mw": round(predictions.get('molecular_weight', 0), 1),
                "logp": round(predictions.get('logp', 0), 2),
                "warnings": (warnings or [])[:2]
            },
            summary=f"Drug-like: {predictions.get('is_drug_like', False)}, QED: {predictions.get('qed_score', 0):.2f}"
        )
    
    def add_docking_result(
        self,
        smiles: str,
        target_protein: str,
        binding_affinity: float,
        interactions: Dict[str, Any],
        confidence: str = ""
    ) -> None:
        """Store docking results"""
        self.current_target = target_protein
        
        self.results[WorkflowType.DOCKING] = WorkflowResult(
            workflow_type=WorkflowType.DOCKING,
            data={
                "smiles": smiles[:50],
                "target": target_protein,
                "affinity_kcal": round(binding_affinity, 2),
                "confidence": confidence,
                "h_bonds": interactions.get('hydrogen_bonds', 0),
                "hydrophobic": interactions.get('hydrophobic_contacts', 0),
                "ionic": interactions.get('ionic_interactions', 0)
            },
            summary=f"{target_protein}: {binding_affinity:.2f} kcal/mol ({confidence})"
        )
    
    def add_chemical_space_result(
        self,
        molecules_analyzed: int,
        clusters: List[Dict[str, Any]],
        diversity_metrics: Dict[str, float]
    ) -> None:
        """Store chemical space results"""
        self.results[WorkflowType.CHEMICAL_SPACE] = WorkflowResult(
            workflow_type=WorkflowType.CHEMICAL_SPACE,
            data={
                "molecules": molecules_analyzed,
                "clusters": len(clusters),
                "similarity": round(diversity_metrics.get('avg_similarity', 0), 3),
                "diversity": round(diversity_metrics.get('diversity_score', 0), 3),
                "coverage": round(diversity_metrics.get('coverage', 0), 3)
            },
            summary=f"{molecules_analyzed} molecules in {len(clusters)} clusters"
        )
    
    # ==================== Optimized Context Retrieval ====================
    
    def get_full_context(self, max_chars: int = 1200) -> str:
        """Get compact context string for LLM"""
        if not self.results and not self.current_molecule and not self.current_target:
            # No results stored — return empty so the model uses general knowledge
            return ""
        
        parts = []
        
        if self.current_molecule:
            parts.append(f"Current: {self.current_molecule}")
        if self.current_target:
            parts.append(f"Target: {self.current_target}")
        
        for wt, result in self.results.items():
            parts.append(f"\n[{wt.value}]")
            for key, value in result.data.items():
                if isinstance(value, dict):
                    parts.append(f"  {key}: {json.dumps(value)}")
                elif isinstance(value, list):
                    parts.append(f"  {key}: {', '.join(str(v) for v in value[:3])}")
                else:
                    parts.append(f"  {key}: {value}")
        
        context = '\n'.join(parts)
        return context[:max_chars]
    
    def get_workflow_context(self, workflow_type: WorkflowType) -> str:
        """Get context for specific workflow"""
        if workflow_type not in self.results:
            return ""
        
        result = self.results[workflow_type]
        lines = [f"[{workflow_type.value}] {result.summary}"]
        
        for key, value in result.data.items():
            if isinstance(value, dict):
                lines.append(f"  {key}: {json.dumps(value)}")
            elif isinstance(value, list):
                lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines)
    
    def set_current_molecule(self, name: str, smiles: str, properties: Dict = None) -> None:
        """Set current molecule"""
        self.current_molecule = smiles[:50]
    
    def clear_results(self, workflow_type: Optional[WorkflowType] = None) -> None:
        """Clear results"""
        if workflow_type:
            self.results.pop(workflow_type, None)
        else:
            self.results.clear()
    
    def get_summary(self) -> Dict[str, int]:
        """Get count of stored results"""
        return {wt.value: 1 if wt in self.results else 0 for wt in WorkflowType}
