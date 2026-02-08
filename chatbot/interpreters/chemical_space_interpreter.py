"""
Chemical Space Interpreter

Interprets chemical space analysis, clustering, and diversity results.
"""

from typing import Dict, Any, List
from .base_interpreter import BaseInterpreter


class ChemicalSpaceInterpreter(BaseInterpreter):
    """Interpreter for chemical space analysis results"""
    
    @property
    def workflow_type(self) -> str:
        return "chemical_space"
    
    @property
    def keywords(self) -> List[str]:
        return [
            'chemical space', 'cluster', 'diversity', 'similarity', 'tanimoto',
            'fingerprint', 'pca', 'tsne', 't-sne', 'scaffold', 'analog',
            'novelty', 'exploration', 'coverage'
        ]
    
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format chemical space results for LLM context"""
        parts = ["## Chemical Space Analysis\n"]
        
        # Overview
        parts.append(f"**Molecules Analyzed:** {data.get('molecules_count', 0)}")
        parts.append(f"**Number of Clusters:** {data.get('num_clusters', 0)}")
        
        # Diversity metrics
        metrics = data.get('diversity_metrics', {})
        if metrics:
            parts.append(f"\n**Diversity Metrics:**")
            parts.append(f"- Average Tanimoto Similarity: {self.format_metric(metrics.get('avg_similarity', 0))}")
            parts.append(f"- Diversity Score: {self.format_metric(metrics.get('diversity_score', 0))}")
            parts.append(f"- Scaffold Diversity: {self.format_metric(metrics.get('scaffold_diversity', 0))}")
            parts.append(f"- Coverage: {self.format_metric(metrics.get('coverage', 0) * 100)}%")
        
        # Clusters
        clusters = data.get('clusters', [])
        if clusters:
            parts.append(f"\n**Cluster Summary:**")
            for i, cluster in enumerate(clusters[:5], 1):
                size = cluster.get('size', 0)
                centroid = cluster.get('centroid_smiles', 'N/A')[:30]
                parts.append(f"{i}. Cluster: {size} molecules, centroid: {centroid}...")
        
        # Interpretation
        avg_sim = metrics.get('avg_similarity', 0) if metrics else 0
        if avg_sim > 0:
            if avg_sim > 0.7:
                parts.append(f"\n**Assessment:** High similarity between molecules - consider diversifying 🟡")
            elif avg_sim > 0.4:
                parts.append(f"\n**Assessment:** Moderate diversity - good balance of exploration 🟢")
            else:
                parts.append(f"\n**Assessment:** High diversity - broad chemical space coverage 🟢")
        
        return "\n".join(parts)
    
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        metrics = data.get('diversity_metrics', {})
        
        return {
            "Molecules": data.get('molecules_count', 0),
            "Clusters": data.get('num_clusters', 0),
            "Avg Similarity": self.format_metric(metrics.get('avg_similarity', 0)),
            "Diversity": self.format_metric(metrics.get('diversity_score', 0)),
            "Coverage": f"{self.format_metric(metrics.get('coverage', 0) * 100)}%"
        }
