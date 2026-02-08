"""
Multi-Agent Interpreter

Interprets results from the Multi-Agent Drug Discovery Platform.
"""

from typing import Dict, Any, List
from .base_interpreter import BaseInterpreter


class MultiAgentInterpreter(BaseInterpreter):
    """Interpreter for multi-agent coordination results"""
    
    @property
    def workflow_type(self) -> str:
        return "multi_agent"
    
    @property
    def keywords(self) -> List[str]:
        return [
            'agent', 'multi-agent', 'multiagent', 'orchestrator', 
            'collaboration', 'designer', 'specialist', 'validation',
            'critic', 'consensus', 'autonomous'
        ]
    
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format multi-agent results for LLM context"""
        parts = ["## Multi-Agent Drug Discovery Analysis\n"]
        
        # Agent results
        agents = data.get('agents', {})
        if agents:
            parts.append("**Agent Contributions:**")
            for agent_name, result in agents.items():
                status = result.get('status', 'unknown')
                score = result.get('score', 0)
                output = result.get('output', 'No output')
                
                status_icon = "✅" if status == 'complete' else "⏳" if status == 'pending' else "⚠️"
                parts.append(f"\n{status_icon} **{agent_name.replace('_', ' ').title()}**")
                parts.append(f"   - Status: {status}")
                parts.append(f"   - Score: {self.format_metric(score)}")
                parts.append(f"   - Output: {output[:100]}...")
        
        # Task summary
        task = data.get('task_summary', {})
        if task:
            parts.append(f"\n**Task Summary:**")
            parts.append(f"- Task ID: {task.get('task_id', 'N/A')[-8:]}")
            parts.append(f"- Execution Time: {self.format_metric(task.get('total_execution_time_seconds', 0))}s")
        
        # Recommendations
        recs = data.get('recommendations', [])
        if recs:
            parts.append(f"\n**Recommendations:**")
            for i, rec in enumerate(recs[:5], 1):
                parts.append(f"{i}. {rec}")
        
        return "\n".join(parts)
    
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        agents = data.get('agents', {})
        completed = sum(1 for a in agents.values() if a.get('status') == 'complete')
        
        return {
            "Agents Used": len(agents),
            "Completed": f"{completed}/{len(agents)}",
            "Execution Time": f"{data.get('task_summary', {}).get('total_execution_time_seconds', 0):.1f}s",
            "Recommendations": len(data.get('recommendations', []))
        }
