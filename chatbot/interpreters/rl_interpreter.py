"""
RL Generation Interpreter

Interprets results from the Improved RL Molecular Generator.
"""

from typing import Dict, Any, List
from .base_interpreter import BaseInterpreter


class RLGenerationInterpreter(BaseInterpreter):
    """Interpreter for RL molecule generation results"""
    
    @property
    def workflow_type(self) -> str:
        return "rl_generation"
    
    @property
    def keywords(self) -> List[str]:
        return [
            'rl', 'reinforcement', 'generation', 'generated', 'reward',
            'training', 'molecule generation', 'optimize', 'best molecule',
            'drug-likeness', 'qed', 'novelty', 'validity'
        ]
    
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format RL generation results for LLM context"""
        parts = ["## RL Molecule Generation Results\n"]
        
        # Training stats
        stats = data.get('training_stats', {})
        if stats:
            parts.append(f"**Training Statistics:**")
            parts.append(f"- Total generations: {stats.get('total_generations', 'N/A')}")
            parts.append(f"- Best reward achieved: {self.format_metric(stats.get('best_reward', 0))}")
            parts.append(f"- Valid molecules rate: {self.format_metric(stats.get('valid_rate', 0) * 100)}%")
        
        # Best molecule
        best = data.get('best_molecule', {})
        if best:
            parts.append(f"\n**Best Generated Molecule:**")
            parts.append(f"- SMILES: {best.get('smiles', 'N/A')}")
            parts.append(f"- Total Reward: {self.format_metric(best.get('total_reward', 0))}")
            
            rewards = best.get('rewards', best.get('scores', {}))
            if rewards:
                parts.append("- Reward Components:")
                parts.append(f"  - Binding Affinity: {self.format_metric(rewards.get('binding_affinity', 0))}")
                parts.append(f"  - Drug-likeness: {self.format_metric(rewards.get('drug_likeness', 0))}")
                parts.append(f"  - Novelty: {self.format_metric(rewards.get('novelty', 0))}")
                parts.append(f"  - Validity: {self.format_metric(rewards.get('validity', 0))}")
        
        # Summary
        parts.append(f"\n**Summary:** Generated {data.get('molecules_count', 0)} molecules")
        if data.get('target_protein'):
            parts.append(f"Target protein: {data.get('target_protein')}")
        
        return "\n".join(parts)
    
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        best = data.get('best_molecule', {})
        stats = data.get('training_stats', {})
        
        return {
            "Molecules Generated": data.get('molecules_count', 0),
            "Best Reward": self.format_metric(best.get('total_reward', stats.get('best_reward', 0))),
            "Target Protein": data.get('target_protein', 'N/A'),
            "Valid Rate": f"{self.format_metric(stats.get('valid_rate', 0) * 100)}%"
        }
