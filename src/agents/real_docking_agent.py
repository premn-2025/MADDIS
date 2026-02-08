import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDockingAgent:
    """
    Autonomous agent for real molecular docking using AutoDock Vina

    Performs physics-based docking simulations to predict:
        - Binding poses (HOW molecule binds)
        - Binding affinity (HOW STRONG)
        - Protein-ligand interactions
    """

    def __init__(self, protein_library_path: str = "data/proteins"):
        """
        Initialize docking agent

        Args:
            protein_library_path: Directory containing PDB protein structures
        """
        self.agent_id = "real_docking_agent"
        self.protein_library_path = Path(protein_library_path)
        self.protein_library_path.mkdir(parents=True, exist_ok=True)

        # Vina instance (will be initialized per docking)
        self.vina = None

        # Protein structures cache
        self.proteins = {}

        logger.info(f"Real Docking Agent initialized")
        logger.info(f"Protein library: {self.protein_library_path}")

    def smiles_to_3d_conformer(self, smiles: str, optimize: bool = True) -> Optional[Chem.Mol]:
        """
        Convert SMILES to 3D molecule with embedded coordinates

        Args:
            smiles: SMILES string
            optimize: Whether to optimize with MMFF force field

        Returns:
            RDKit molecule with 3D coordinates
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                logger.warning(f"Failed to embed molecule, trying with random coords")
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

            # Optimize geometry
            if optimize:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                except:
                    logger.warning("MMFF optimization failed, using UFF")
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)

            return mol

        except Exception as e:
            logger.error(f"Error converting SMILES to 3D: {e}")
            return None

    def mol_to_pdbqt(self, mol: Chem.Mol, output_path: str) -> bool:
        """
        Convert RDKit molecule to PDBQT format for Vina

        Args:
            mol: RDKit molecule with 3D coordinates
            output_path: Where to save PDBQT file

        Returns:
            Success status
        """
        try:
            # Prepare molecule with Meeko
            preparator = MoleculePreparation()
            preparator.prepare(mol)

            # Write PDBQT
            pdbqt_string = preparator.write_pdbqt_string()

            with open(output_path, 'w') as f:
                f.write(pdbqt_string)

            return True

        except Exception as e:
            logger.error(f"Error converting to PDBQT: {e}")
            return False

    def prepare_ligand(self, smiles: str, output_path: str) -> bool:
        """
        Full pipeline: SMILES → 3D → PDBQT

        Args:
            smiles: SMILES string
            output_path: Where to save ligand PDBQT

        Returns:
            Success status
        """
        # Convert to 3D
        mol_3d = self.smiles_to_3d_conformer(smiles, optimize=True)
        if mol_3d is None:
            return False

        # Convert to PDBQT
        return self.mol_to_pdbqt(mol_3d, output_path)

    def detect_binding_site(self, protein_path: str) -> Tuple[List[float], List[float]]:
        """
        Detect binding site from protein structure

        For now, uses center of mass. In production, would use:
            - Fpocket for pocket detection
            - Known ligand position from PDB
            - User-specified coordinates

        Args:
            protein_path: Path to protein PDB file

        Returns:
            center: [x, y, z] coordinates of binding site
            box_size: [x, y, z] search box dimensions
        """
        try:
            # Read protein coordinates
            coords = []
            with open(protein_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coords.append([x, y, z])
                        except:
                            continue

            if not coords:
                logger.warning("No coordinates found, using default binding site")
                return [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]

            coords = np.array(coords)

            # Center of mass as approximate binding site
            center = coords.mean(axis=0).tolist()

            # Box size: cover protein with some padding
            ranges = coords.max(axis=0) - coords.min(axis=0)
            box_size = (ranges * 0.6).tolist()  # 60% of protein size
            box_size = [max(15.0, min(s, 25.0)) for s in box_size]  # Clamp 15-25 Å

            logger.info(f"Binding site detected: center={center}, box={box_size}")

            return center, box_size

        except Exception as e:
            logger.error(f"Error detecting binding site: {e}")
            return [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]

    async def dock_molecule(
        self,
        smiles: str,
        protein_name: str,
        protein_path: Optional[str] = None,
        center: Optional[List[float]] = None,
        box_size: Optional[List[float]] = None,
        exhaustiveness: int = 8,
        n_poses: int = 9
    ) -> Dict:
        """
        Perform molecular docking with AutoDock Vina

        Args:
            smiles: Ligand SMILES string
            protein_name: Target protein identifier
            protein_path: Path to protein PDB/PDBQT file (optional)
            center: Binding site center [x,y,z] (optional, auto-detect if None)
            box_size: Search box size [x,y,z] (optional, auto-detect if None)
            exhaustiveness: Vina search exhaustiveness (8-32, higher=better/slower)
            n_poses: Number of binding poses to generate

        Returns:
            Dictionary with docking results
        """
        logger.info(f"Starting docking: {smiles} → {protein_name}")

        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Prepare ligand
                ligand_pdbqt = tmpdir / "ligand.pdbqt"
                logger.info("Preparing ligand...")
                if not self.prepare_ligand(smiles, str(ligand_pdbqt)):
                    return {
                        'success': False,
                        'error': 'Failed to prepare ligand',
                        'smiles': smiles,
                        'protein': protein_name
                    }

                # Get protein file
                if protein_path is None:
                    protein_path = self.protein_library_path / f"{protein_name}.pdbqt"
                    if not protein_path.exists():
                        return {
                            'success': False,
                            'error': f'Protein {protein_name} not found in library',
                            'smiles': smiles,
                            'protein': protein_name
                        }

                # Auto-detect binding site if not provided
                if center is None or box_size is None:
                    logger.info("Auto-detecting binding site...")
                    center, box_size = self.detect_binding_site(str(protein_path))

                # Initialize Vina
                logger.info("Initializing AutoDock Vina...")
                v = Vina(sf_name='vina', cpu=4, verbosity=0)

                # Set receptor
                v.set_receptor(str(protein_path))

                # Set ligand
                v.set_ligand_from_file(str(ligand_pdbqt))

                # Set search space
                v.compute_vina_maps(center=center, box_size=box_size)

                # Dock!
                logger.info(f"Docking with exhaustiveness={exhaustiveness}...")
                v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

                # Get results
                energies = v.energies(n_poses=n_poses)

                # Extract best pose
                best_energy = energies[0][0] if energies else None

                # Save docked poses
                output_pdbqt = tmpdir / "docked.pdbqt"
                v.write_poses(str(output_pdbqt), n_poses=n_poses)

                # Read docked structure
                with open(output_pdbqt, 'r') as f:
                    docked_pdbqt = f.read()

                logger.info(f"Docking complete! Best affinity: {best_energy:.2f} kcal/mol")

                return {
                    'success': True,
                    'smiles': smiles,
                    'protein': protein_name,
                    'binding_affinity': best_energy,
                    'binding_affinity_unit': 'kcal/mol',
                    'n_poses': len(energies),
                    'all_energies': [e[0] for e in energies],
                    'center': center,
                    'box_size': box_size,
                    'exhaustiveness': exhaustiveness,
                    'docked_structure': docked_pdbqt,
                    'method': 'AutoDock Vina'
                }

        except Exception as e:
            logger.error(f"Docking failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'smiles': smiles,
                'protein': protein_name
            }

    async def batch_dock(
        self,
        smiles_list: List[str],
        protein_name: str,
        max_concurrent: int = 4
    ) -> List[Dict]:
        """
        Dock multiple molecules in parallel

        Args:
            smiles_list: List of SMILES strings
            protein_name: Target protein
            max_concurrent: Maximum parallel dockings

        Returns:
            List of docking results
        """
        logger.info(f"Batch docking {len(smiles_list)} molecules → {protein_name}")

        # Create semaphore to limit concurrent dockings
        semaphore = asyncio.Semaphore(max_concurrent)

        async def dock_with_semaphore(smiles):
            async with semaphore:
                return await self.dock_molecule(smiles, protein_name)

        # Dock all molecules
        tasks = [dock_with_semaphore(smiles) for smiles in smiles_list]
        results = await asyncio.gather(*tasks)

        # Summary statistics
        successful = [r for r in results if r.get('success')]
        logger.info(f"Batch complete: {len(successful)}/{len(smiles_list)} successful")

        if successful:
            affinities = [r['binding_affinity'] for r in successful]
            logger.info(f"Affinity range: {min(affinities):.2f} to {max(affinities):.2f} kcal/mol")
            logger.info(f"Mean affinity: {np.mean(affinities):.2f} kcal/mol")

        return results

    def calculate_rmsd(self, mol1_coords: np.ndarray, mol2_coords: np.ndarray) -> float:
        """
        Calculate RMSD between two sets of coordinates

        Args:
            mol1_coords: First molecule coordinates (N, 3)
            mol2_coords: Second molecule coordinates (N, 3)

        Returns:
            RMSD in Angstroms
        """
        if mol1_coords.shape != mol2_coords.shape:
            raise ValueError("Coordinate arrays must have same shape")

        diff = mol1_coords - mol2_coords
        rmsd = np.sqrt((diff ** 2).sum() / len(mol1_coords))

        return rmsd

    def validate_docking(
        self,
        known_ligand_smiles: str,
        protein_name: str,
        crystal_structure_path: str
    ) -> Dict:
        """
        Validate docking by re-docking known inhibitor

        Args:
            known_ligand_smiles: SMILES of crystallized ligand
            protein_name: Target protein
            crystal_structure_path: PDB with ligand

        Returns:
            Validation results with RMSD
        """
        logger.info("Validating docking accuracy...")

        # TODO: Implement validation
        # 1. Extract ligand coordinates from crystal structure
        # 2. Re-dock ligand
        # 3. Calculate RMSD between docked and crystal pose
        # 4. RMSD < 2.0 Å = good, < 3.0 Å = acceptable

        return {
            'validation': 'not_implemented',
            'message': 'Use validate_docking_full() for complete validation'
        }

    async def analyze_interactions(self, docking_result: Dict) -> Dict:
        """
        Analyze protein-ligand interactions from docked pose

        Args:
            docking_result: Result from dock_molecule()

        Returns:
            Interaction analysis (H-bonds, hydrophobic, etc.)
        """
        # TODO: Implement interaction analysis
        # - Parse PDBQT for contacts
        # - Identify H-bonds (distance + angle criteria)
        # - Identify hydrophobic interactions
        # - Identify pi-pi stacking
        # - Identify salt bridges

        return {
            'analysis': 'not_implemented',
            'message': 'Interaction analysis coming in next version'
        }


# Example usage and testing
async def test_real_docking():
    """Test the real docking agent"""

    print("=" * 80)
    print("TESTING REAL MOLECULAR DOCKING AGENT")
    print("=" * 80 + "\n")

    # Initialize agent
    agent = RealDockingAgent()

    # Test SMILES to 3D conversion
    print("Test 1: SMILES to 3D conversion")
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    mol_3d = agent.smiles_to_3d_conformer(test_smiles)

    if mol_3d:
        print(f"Successfully converted aspirin to 3D")
        print(f"Atoms: {mol_3d.GetNumAtoms()}")
        print(f"Conformers: {mol_3d.GetNumConformers()}")
    else:
        print("Failed to convert SMILES")

    print("\n" + "-" * 80 + "\n")

    # Test ligand preparation
    print("Test 2: Ligand preparation (SMILES → PDBQT)")
    with tempfile.TemporaryDirectory() as tmpdir:
        ligand_path = Path(tmpdir) / "test_ligand.pdbqt"
        success = agent.prepare_ligand(test_smiles, str(ligand_path))

        if success:
            print(f"Successfully prepared ligand")
            print(f"Output: {ligand_path.name}")
            print(f"Size: {ligand_path.stat().st_size} bytes")
        else:
            print("Failed to prepare ligand")

    print("\n" + "-" * 80 + "\n")

    print("Full docking test requires protein structures")
    print("Run download_proteins.py to get PDB files")
    print("Then use agent.dock_molecule() for real docking")

    print("\n" + "=" * 80)
    print("REAL DOCKING AGENT TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_real_docking())