"""
Docking Interpreter

Interprets molecular docking and binding affinity results.
"""

from typing import Dict, Any, List
from .base_interpreter import BaseInterpreter


class DockingInterpreter(BaseInterpreter):
    """Interpreter for molecular docking results"""
    
    @property
    def workflow_type(self) -> str:
        return "docking"
    
    @property
    def keywords(self) -> List[str]:
        return [
            'docking', 'binding', 'affinity', 'kcal', 'pose', 'interaction',
            'hydrogen bond', 'h-bond', 'hydrophobic', 'protein', 'ligand',
            'vina', 'autodock', 'pocket', 'binding site'
        ]
    
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format docking results for LLM context"""
        parts = ["## Molecular Docking Results\n"]
        
        # Basic info
        parts.append(f"**Ligand SMILES:** {data.get('smiles', 'N/A')}")
        parts.append(f"**Target Protein:** {data.get('target_protein', 'N/A')}")
        
        # Binding affinity
        affinity = data.get('binding_affinity', 0)
        confidence = data.get('confidence', 'Unknown')
        parts.append(f"\n**Binding Affinity:** {self.format_metric(affinity)} kcal/mol")
        parts.append(f"**Confidence:** {confidence}")
        
        # Interpretation
        if affinity:
            if affinity < -10:
                parts.append("**Assessment:** Excellent binding (< -10 kcal/mol) - Very strong affinity 🟢")
            elif affinity < -8:
                parts.append("**Assessment:** Good binding (-8 to -10 kcal/mol) - Strong drug candidate 🟢")
            elif affinity < -6:
                parts.append("**Assessment:** Moderate binding (-6 to -8 kcal/mol) - Potential for optimization 🟡")
            else:
                parts.append("**Assessment:** Weak binding (> -6 kcal/mol) - Needs improvement 🔴")
        
        # Interactions
        interactions = data.get('interactions', {})
        if interactions:
            parts.append(f"\n**Protein-Ligand Interactions:**")
            parts.append(f"- Hydrogen Bonds: {interactions.get('hydrogen_bonds', 0)}")
            parts.append(f"- Hydrophobic Contacts: {interactions.get('hydrophobic_contacts', 0)}")
            parts.append(f"- Ionic Interactions: {interactions.get('ionic_interactions', 0)}")
            parts.append(f"- Pi-Stacking: {interactions.get('pi_stacking', 0)}")
            
            if interactions.get('key_residues'):
                parts.append(f"- Key Residues: {', '.join(interactions.get('key_residues', []))}")
        
        return "\n".join(parts)
    
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        interactions = data.get('interactions', {})
        
        return {
            "Target": data.get('target_protein', 'N/A'),
            "Binding Affinity": f"{self.format_metric(data.get('binding_affinity', 0))} kcal/mol",
            "Confidence": data.get('confidence', 'N/A'),
            "H-Bonds": interactions.get('hydrogen_bonds', 0),
            "Hydrophobic": interactions.get('hydrophobic_contacts', 0)
        }
