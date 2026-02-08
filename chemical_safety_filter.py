#!/usr/bin/env python3
"""
Chemical Safety Filter - Reject molecules with reactive/toxic groups
"""

from rdkit import Chem
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ChemicalSafetyFilter:
    """Filter out molecules with problematic functional groups"""
    
    def __init__(self):
        # PAINS (Pan Assay Interference Compounds) patterns
        self.reactive_patterns = {
            # Highly reactive groups
            'acid_chloride': 'C(=O)Cl',
            'acyl_halide': 'C(=O)[Cl,Br,I]',
            'sulfonyl_chloride': 'S(=O)(=O)Cl',
            'isocyanate': 'N=C=O',
            'isothiocyanate': 'N=C=S',
            
            # Unstable groups
            'peroxide': 'OO',
            'trioxide': 'OOO',
            'diazo': 'C=[N+]=[N-]',
            'azide': 'N=[N+]=[N-]',
            
            # Reactive aldehydes and ketones (problematic in some contexts)
            'aldehyde': '[CH]=O',
            
            # Strained rings
            'epoxide_small': 'C1OC1',
            
            # Anhydrides
            'anhydride': 'C(=O)OC(=O)',
            
            # Michael acceptors (can be toxic)
            'alpha_beta_unsaturated_carbonyl': 'C=CC=O',
            
            # Thiols (can be reactive)
            'thiol': '[SH]',
            
            # Nitro groups (can be mutagenic)
            'nitro': '[N+](=O)[O-]',
            
            # Quaternary ammonium (often toxic)
            'quat_amine': '[N+]([C,H])([C,H])([C,H])[C,H]',
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for name, smarts in self.reactive_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self.compiled_patterns[name] = pattern
            except Exception as e:
                logger.warning(f"Could not compile pattern {name}: {e}")
    
    def check_safety(self, smiles: str) -> Tuple[bool, List[str]]:
        """
        Check if molecule is safe (no reactive groups)
        
        Args:
            smiles: SMILES string
        
        Returns:
            (is_safe, list_of_issues)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, ["Invalid SMILES"]
        
        issues = []
        
        for name, pattern in self.compiled_patterns.items():
            if mol.HasSubstructMatch(pattern):
                # Format name nicely
                issue_name = name.replace('_', ' ').title()
                issues.append(issue_name)
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    def filter_molecules(self, smiles_list: List[str], verbose: bool = True) -> List[str]:
        """
        Filter a list of SMILES, keeping only safe molecules
        
        Args:
            smiles_list: List of SMILES strings
            verbose: Print rejected molecules
        
        Returns:
            List of safe SMILES
        """
        safe_molecules = []
        
        for smiles in smiles_list:
            is_safe, issues = self.check_safety(smiles)
            
            if is_safe:
                safe_molecules.append(smiles)
            else:
                if verbose:
                    logger.warning(f"⚠️  REJECTED: {smiles}")
                    logger.warning(f"   Issues: {', '.join(issues)}")
        
        if verbose:
            logger.info(f"✓ Filtered {len(smiles_list)} → {len(safe_molecules)} safe molecules")
            logger.info(f"  Rejection rate: {(len(smiles_list) - len(safe_molecules))/len(smiles_list)*100:.1f}%")
        
        return safe_molecules


def check_reactive_groups(smiles: str) -> bool:
    """
    Quick check if molecule is safe (for reward calculation)
    
    Args:
        smiles: SMILES string
    
    Returns:
        True if safe, False if reactive
    """
    filter_obj = ChemicalSafetyFilter()
    is_safe, _ = filter_obj.check_safety(smiles)
    return is_safe


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test
    filter_obj = ChemicalSafetyFilter()
    
    test_smiles = [
        "CN(CNC(=O)OCl)c1cccnc1",  # Acid chloride - SHOULD REJECT
        "CC(=O)Oc1ccccc1C(=O)OOOC(=O)CC#N",  # Peroxide - SHOULD REJECT
        "CC(=O)NCC(C)c1cccc2ncncc12",  # Safe - SHOULD PASS
        "CCOc1ccc(C(=O)O)cc1",  # Safe - SHOULD PASS
    ]
    
    print("Testing Chemical Safety Filter")
    print("="*60)
    
    for smiles in test_smiles:
        is_safe, issues = filter_obj.check_safety(smiles)
        status = "✅ SAFE" if is_safe else "❌ UNSAFE"
        print(f"\n{status}: {smiles}")
        if issues:
            print(f"  Issues: {', '.join(issues)}")
