"""
Multi-Target RL Interpreter

Interprets results from the Multi-Target RL Generator with Pareto optimization.
"""

from typing import Dict, Any, List
from .base_interpreter import BaseInterpreter


class MultiTargetInterpreter(BaseInterpreter):
    """Interpreter for multi-target RL optimization results"""
    
    @property
    def workflow_type(self) -> str:
        return "multi_target_rl"
    
    @property
    def keywords(self) -> List[str]:
        return [
            'multi-target', 'multitarget', 'pareto', 'objectives', 
            'multiple targets', 'trade-off', 'tradeoff', 'dual target',
            'selectivity', 'polypharmacology', 'front'
        ]
    
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format multi-target results for LLM context"""
        parts = ["## Multi-Target RL Optimization Results\n"]
        
        # Targets
        targets = data.get('targets', [])
        parts.append(f"**Target Proteins:** {', '.join(targets) if targets else 'N/A'}")
        
        # Pareto solutions
        solutions = data.get('top_solutions', [])
        parts.append(f"\n**Pareto-Optimal Solutions:** {data.get('pareto_solutions_count', len(solutions))}")
        
        if solutions:
            parts.append("\n**Top 3 Solutions:**")
            for i, sol in enumerate(solutions[:3], 1):
                parts.append(f"\n{i}. SMILES: {sol.get('smiles', 'N/A')}")
                
                affinities = sol.get('target_affinities', {})
                if affinities:
                    for target, affinity in affinities.items():
                        parts.append(f"   - {target}: {self.format_metric(affinity)} kcal/mol")
                
                parts.append(f"   - QED Score: {self.format_metric(sol.get('qed_score', 0))}")
                parts.append(f"   - Total Reward: {self.format_metric(sol.get('total_reward', 0))}")
        
        # Training history
        history = data.get('training_history', {})
        if history:
            parts.append(f"\n**Training Summary:**")
            parts.append(f"- Generations: {history.get('generations', 'N/A')}")
            parts.append(f"- Final Pareto Front Size: {history.get('pareto_size', 'N/A')}")
        
        return "\n".join(parts)
    
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        solutions = data.get('top_solutions', [])
        best = solutions[0] if solutions else {}
        
        return {
            "Targets": ", ".join(data.get('targets', [])),
            "Pareto Solutions": data.get('pareto_solutions_count', 0),
            "Best Total Reward": self.format_metric(best.get('total_reward', 0)),
            "Best QED": self.format_metric(best.get('qed_score', 0))
        }
