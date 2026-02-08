"""
Property Prediction Interpreter

Interprets ADMET and drug-likeness prediction results.
"""

from typing import Dict, Any, List
from .base_interpreter import BaseInterpreter


class PropertyInterpreter(BaseInterpreter):
    """Interpreter for property prediction (ADMET) results"""
    
    @property
    def workflow_type(self) -> str:
        return "property_prediction"
    
    @property
    def keywords(self) -> List[str]:
        return [
            'property', 'admet', 'toxicity', 'solubility', 'permeability',
            'drug-like', 'druglike', 'lipinski', 'qed', 'absorption',
            'metabolism', 'excretion', 'bbb', 'blood-brain'
        ]
    
    def format_context(self, data: Dict[str, Any]) -> str:
        """Format property prediction results for LLM context"""
        parts = ["## Molecular Property Predictions\n"]
        
        # Molecule info
        parts.append(f"**Molecule SMILES:** {data.get('smiles', 'N/A')}")
        
        # Predictions
        preds = data.get('predictions', {})
        if preds:
            parts.append(f"\n**Drug-Likeness Assessment:**")
            parts.append(f"- Drug-like: {'Yes ✅' if preds.get('is_drug_like') else 'No ❌'}")
            parts.append(f"- QED Score: {self.format_metric(preds.get('qed_score', 0))} (0-1, higher is better)")
            parts.append(f"- Lipinski Violations: {preds.get('lipinski_violations', 'N/A')}")
            
            parts.append(f"\n**ADMET Properties:**")
            parts.append(f"- Toxicity Score: {self.format_metric(preds.get('toxicity_score', 0))} (lower is better)")
            parts.append(f"- Solubility: {self.format_metric(preds.get('solubility', 0))}")
            parts.append(f"- Permeability: {self.format_metric(preds.get('permeability', 0))}")
            parts.append(f"- BBB Permeability: {self.format_metric(preds.get('bbb_permeability', 0))}")
            
            parts.append(f"\n**Molecular Descriptors:**")
            parts.append(f"- Molecular Weight: {self.format_metric(preds.get('molecular_weight', 0))} Da")
            parts.append(f"- LogP: {self.format_metric(preds.get('logp', 0))}")
            parts.append(f"- H-Bond Donors: {preds.get('hbd', 'N/A')}")
            parts.append(f"- H-Bond Acceptors: {preds.get('hba', 'N/A')}")
            parts.append(f"- TPSA: {self.format_metric(preds.get('tpsa', 0))} Å²")
        
        # Warnings
        warnings = data.get('warnings', [])
        if warnings:
            parts.append(f"\n**⚠️ Warnings:**")
            for w in warnings:
                parts.append(f"- {w}")
        
        return "\n".join(parts)
    
    def get_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for display"""
        preds = data.get('predictions', {})
        
        return {
            "Drug-Like": "Yes" if preds.get('is_drug_like') else "No",
            "QED Score": self.format_metric(preds.get('qed_score', 0)),
            "Lipinski Violations": preds.get('lipinski_violations', 'N/A'),
            "Toxicity": self.format_metric(preds.get('toxicity_score', 0)),
            "MW": f"{self.format_metric(preds.get('molecular_weight', 0))} Da"
        }
