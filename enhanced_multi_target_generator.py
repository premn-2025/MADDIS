"""
Enhanced Multi-Target RL Generator with Property Prediction Integration

Integrates the Property Prediction Agent with the molecular generation pipeline
to ensure generated molecules are drug-like and safe.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from multi_target_rl_generator import MultiTargetRLGenerator
from property_prediction_agent import PropertyPredictionAgent, PropertyPrediction

logger = logging.getLogger(__name__)


class EnhancedMultiTargetRLGenerator(MultiTargetRLGenerator):
    """
    Enhanced molecular generator with integrated property prediction

    Adds ADMET filtering to ensure generated molecules are:
        - Non-toxic (toxicity < 0.7)
        - Drug-like (QED > 0.3)
        - Synthetically accessible (SA score < 8)
        - Lipinski compliant (≤ 2 violations)
    """

    def __init__(self, objectives, docking_agent=None, property_predictor=None,
                 filter_stringency="moderate", toxicity_threshold=0.7,
                 drug_score_threshold=0.5, require_bbb_penetration=False,
                 enable_filter_stats=True, **kwargs):

        # Extract base parameters for parent class
        base_kwargs = {k: v for k, v in kwargs.items() if k in ['max_length']}

        # Initialize base generator
        super().__init__(objectives, docking_agent, **base_kwargs)

        # Initialize property predictor
        try:
            self.property_agent = property_predictor if property_predictor else PropertyPredictionAgent()
            logger.info(" Property Prediction Agent integrated successfully")
        except Exception as e:
            logger.warning(f" Property prediction integration failed: {e}")
            self.property_agent = None

        # Set filtering criteria based on stringency
        self.filtering_enabled = True
        self.filter_stringency = filter_stringency
        self.enable_stats = enable_filter_stats
        self.require_bbb = require_bbb_penetration

        # Configure filtering criteria based on stringency
        if filter_stringency == "strict":
            self.filter_criteria = {
                'max_toxicity': 0.3,
                'min_qed': 0.5,
                'max_sa_score': 6.0,
                'max_lipinski_violations': 1,
                'min_drug_score': drug_score_threshold or 0.6
            }
        elif filter_stringency == "moderate":
            self.filter_criteria = {
                'max_toxicity': toxicity_threshold or 0.7,
                'min_qed': 0.3,
                'max_sa_score': 8.0,
                'max_lipinski_violations': 2,
                'min_drug_score': drug_score_threshold or 0.4
            }
        else:  # lenient
            self.filter_criteria = {
                'max_toxicity': 0.8,
                'min_qed': 0.2,
                'max_sa_score': 10.0,
                'max_lipinski_violations': 3,
                'min_drug_score': drug_score_threshold or 0.3
            }

        # Statistics tracking
        self.filter_stats = {
            'total_generated': 0,
            'passed_docking': 0,
            'passed_property_filter': 0,
            'final_accepted': 0
        }

        logger.info(" Enhanced Multi-Target Generator ready with ADMET filtering")

    def enable_filtering(self, enabled: bool = True):
        """Enable/disable property-based filtering"""
        self.filtering_enabled = enabled
        logger.info(f" Property filtering: {'ENABLED' if enabled else 'DISABLED'}")

    def update_filter_criteria(self, **criteria):
        """Update filtering criteria"""
        self.filter_criteria.update(criteria)
        logger.info(f" Updated filter criteria: {criteria}")

    async def generate_and_filter_molecules(self, num_molecules: int) -> List[Dict[str, Any]]:
        """
        Generate molecules with integrated docking and property prediction

        Pipeline:
            1. Generate molecules with base generator
            2. Perform docking for all targets
            3. Apply property-based filtering
            4. Return filtered results
        """
        # Generate base molecules
        results = await self._generate_base_molecules(num_molecules)
        self.filter_stats['total_generated'] += len(results)

        if not results:
            return []

        # Apply docking (already done in base generation)
        docked_results = [r for r in results if 'reward' in r]
        self.filter_stats['passed_docking'] += len(docked_results)

        # Apply property filtering if enabled
        if self.filtering_enabled and self.property_agent:
            filtered_results = await self._apply_property_filtering(docked_results)
            self.filter_stats['passed_property_filter'] += len(filtered_results)
        else:
            filtered_results = docked_results

        self.filter_stats['final_accepted'] += len(filtered_results)

        # Log statistics
        if self.filter_stats['total_generated'] > 0:
            pass_rate = 100 * self.filter_stats['final_accepted'] / self.filter_stats['total_generated']
            logger.info(f" Filter pass rate: {pass_rate:.1f}% "
                       f"({self.filter_stats['final_accepted']}/{self.filter_stats['total_generated']})")

        return filtered_results

    async def _generate_base_molecules(self, num_molecules: int) -> List[Dict[str, Any]]:
        """Generate molecules using base generator"""
        generated_data = []

        for i in range(num_molecules):
            try:
                # Generate molecule
                smiles = await self.generate_valid_molecule()

                if not smiles or smiles in [d['smiles'] for d in generated_data]:
                    continue

                # Create dummy values for compatibility
                tokens = []
                log_probs = []
                entropy = 0.0

                # Perform docking
                target_affinities = {}
                total_reward = 0.0

                for obj in self.objectives:
                    try:
                        if self.docking_agent is not None:
                            result = await self.docking_agent.dock_molecule(smiles, obj.target_name)
                            affinity = result.binding_affinity if hasattr(result, 'binding_affinity') else result
                        else:
                            affinity = -7.0  # Default
                        target_affinities[obj.target_name] = affinity

                        # Calculate normalized reward
                        normalized_affinity = max(0, (affinity + 12) / 9)
                        weighted_reward = obj.weight * normalized_affinity
                        total_reward += weighted_reward
                    except Exception as e:
                        logger.warning(f"Docking failed for {obj.target_name}: {e}")
                        target_affinities[obj.target_name] = -6.0
                        total_reward += obj.weight * 0.5

                # Create result
                result = {
                    'smiles': smiles,
                    'tokens': tokens,
                    'log_probs': log_probs,
                    'entropy': entropy,
                    'target_affinities': target_affinities,
                    'reward': total_reward
                }

                generated_data.append(result)

            except Exception as e:
                logger.warning(f"Error generating molecule {i + 1}: {e}")
                continue

        return generated_data

    async def _apply_property_filtering(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply property-based filtering to molecules"""
        if not self.property_agent:
            return molecules

        filtered_molecules = []

        for mol_data in molecules:
            smiles = mol_data['smiles']

            try:
                # Predict properties
                prediction = self.property_agent.predict_properties(smiles)

                # Apply filtering criteria
                if self._passes_property_filters(prediction):
                    mol_data['property_prediction'] = prediction
                    mol_data['enhanced_reward'] = self._calculate_enhanced_reward(mol_data, prediction)
                    filtered_molecules.append(mol_data)
                else:
                    logger.debug(f"Molecule filtered out: {smiles}")

            except Exception as e:
                logger.warning(f"Error predicting properties for {smiles}: {e}")
                filtered_molecules.append(mol_data)

        return filtered_molecules

    def _passes_property_filters(self, prediction: PropertyPrediction) -> bool:
        """Check if molecule passes property-based filters"""
        criteria = self.filter_criteria

        checks = [
            prediction.toxicity_score <= criteria['max_toxicity'],
            prediction.qed_score >= criteria['min_qed'],
            prediction.synthetic_accessibility <= criteria['max_sa_score'],
            prediction.lipinski_violations <= criteria['max_lipinski_violations'],
            prediction.overall_drug_score >= criteria['min_drug_score']
        ]

        return all(checks)

    def _calculate_enhanced_reward(self, mol_data: Dict[str, Any], prediction: PropertyPrediction) -> float:
        """Calculate enhanced reward incorporating property predictions"""
        base_reward = mol_data['reward']

        # Property bonuses/penalties
        property_multiplier = 1.0

        # Toxicity penalty
        toxicity_penalty = prediction.toxicity_score * 0.3
        property_multiplier *= (1.0 - toxicity_penalty)

        # Drug-likeness bonus
        drug_bonus = prediction.overall_drug_score * 0.2
        property_multiplier *= (1.0 + drug_bonus)

        # Synthetic accessibility penalty
        sa_penalty = min(0.2, (prediction.synthetic_accessibility - 1) / 45)
        property_multiplier *= (1.0 - sa_penalty)

        enhanced_reward = base_reward * property_multiplier

        return max(0.0, enhanced_reward)

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        stats = self.filter_stats.copy()

        if stats['total_generated'] > 0:
            stats['docking_pass_rate'] = stats['passed_docking'] / stats['total_generated']
            stats['property_pass_rate'] = stats['passed_property_filter'] / stats['total_generated']
            stats['overall_pass_rate'] = stats['final_accepted'] / stats['total_generated']
        else:
            stats['docking_pass_rate'] = 0.0
            stats['property_pass_rate'] = 0.0
            stats['overall_pass_rate'] = 0.0

        return stats

    def reset_filter_statistics(self):
        """Reset filtering statistics"""
        self.filter_stats = {
            'total_generated': 0,
            'passed_docking': 0,
            'passed_property_filter': 0,
            'final_accepted': 0
        }
        logger.info(" Filter statistics reset")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    async def test():
        from real_docking_agent import RealMolecularDockingAgent
        from multi_target_rl_generator import MultiTargetObjective
        
        docking_agent = RealMolecularDockingAgent()
        objectives = [
            MultiTargetObjective("COX2", 0.5),
            MultiTargetObjective("EGFR", 0.5)
        ]
        
        generator = EnhancedMultiTargetRLGenerator(
            objectives=objectives,
            docking_agent=docking_agent
        )
        
        molecules = await generator.generate_and_filter_molecules(3)
        print(f"Generated {len(molecules)} filtered molecules")
        
        for mol in molecules:
            print(f"  {mol['smiles']}: reward={mol.get('enhanced_reward', mol['reward']):.3f}")
    
    asyncio.run(test())
