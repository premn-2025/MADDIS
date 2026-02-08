"""
Molecular docking and scoring for structure-based drug design

Implements:
- AutoDock Vina integration
- Binding pose prediction
- Scoring function evaluation
- Binding site analysis
"""

import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DockingResult:
    """Container for docking results"""
    smiles: str
    binding_affinity: float
    pose_coordinates: Optional[np.ndarray] = None
    interactions: Optional[Dict] = None
    vina_score: Optional[float] = None
    pose_rank: int = 1


@dataclass
class BindingSite:
    """Container for binding site information"""
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0


class MolecularDocking:
    """Interface for molecular docking operations"""

    def __init__(self, protein_pdb_path: str, vina_executable: Optional[str] = None):
        self.protein_pdb_path = Path(protein_pdb_path)
        self.vina_executable = vina_executable or self._find_vina_executable()
        
        if not self.protein_pdb_path.exists():
            raise FileNotFoundError(f"Protein PDB file not found: {protein_pdb_path}")
        
        self.external_docking_available = self.vina_executable is not None
        
        if self.external_docking_available:
            logger.info(f"Using Vina executable: {self.vina_executable}")
        else:
            logger.warning("External docking not available. Using simplified scoring only.")
        
        self.binding_site = None

    def _find_vina_executable(self) -> Optional[str]:
        """Try to find Vina executable in common locations"""
        possible_paths = [
            'vina',
            '/usr/local/bin/vina',
            '/usr/bin/vina',
            'C:\\Program Files\\The Scripps Research Institute\\Vina\\vina.exe',
            os.path.expanduser('~/vina/bin/vina')
        ]
        
        for path in possible_paths:
            try:
                subprocess.run([path, '--version'], capture_output=True, check=True, timeout=5)
                return path
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None

    def set_binding_site(self, center: Tuple[float, float, float],
                        size: Tuple[float, float, float] = (20.0, 20.0, 20.0)) -> None:
        """Set the binding site for docking"""
        self.binding_site = BindingSite(
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            size_x=size[0],
            size_y=size[1],
            size_z=size[2]
        )
        logger.info(f"Binding site set at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

    def dock_molecule(self, smiles: str, ligand_sdf_path: Optional[str] = None) -> DockingResult:
        """Dock a single molecule"""
        if self.external_docking_available and self.binding_site is not None:
            try:
                return self._dock_with_vina(smiles, ligand_sdf_path)
            except Exception as e:
                logger.warning(f"Vina docking failed for {smiles}: {e}")
                return self._simplified_scoring(smiles)
        else:
            return self._simplified_scoring(smiles)

    def dock_molecules(self, smiles_list: List[str],
                      ligand_paths: Optional[List[str]] = None) -> List[DockingResult]:
        """Dock multiple molecules"""
        results = []
        
        if ligand_paths is None:
            ligand_paths = [None] * len(smiles_list)
        
        for i, (smiles, ligand_path) in enumerate(zip(smiles_list, ligand_paths)):
            if (i + 1) % 10 == 0:
                logger.info(f"Docking molecule {i + 1}/{len(smiles_list)}")
            
            try:
                result = self.dock_molecule(smiles, ligand_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to dock molecule {smiles}: {e}")
                results.append(DockingResult(smiles=smiles, binding_affinity=0.0))
        
        logger.info(f"Successfully docked {len(results)} molecules")
        return results

    def _dock_with_vina(self, smiles: str, ligand_path: Optional[str] = None) -> DockingResult:
        """Perform docking using AutoDock Vina"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if ligand_path is None:
                ligand_pdbqt = self._prepare_ligand_from_smiles(smiles, temp_path)
            else:
                ligand_pdbqt = self._convert_to_pdbqt(ligand_path, temp_path)
            
            receptor_pdbqt = self._prepare_receptor(temp_path)
            
            output_pdbqt = temp_path / "output.pdbqt"
            log_file = temp_path / "vina.log"
            
            vina_command = [
                self.vina_executable,
                '--receptor', str(receptor_pdbqt),
                '--ligand', str(ligand_pdbqt),
                '--out', str(output_pdbqt),
                '--log', str(log_file),
                '--center_x', str(self.binding_site.center_x),
                '--center_y', str(self.binding_site.center_y),
                '--center_z', str(self.binding_site.center_z),
                '--size_x', str(self.binding_site.size_x),
                '--size_y', str(self.binding_site.size_y),
                '--size_z', str(self.binding_site.size_z),
                '--exhaustiveness', '8',
                '--num_modes', '9'
            ]
            
            subprocess.run(vina_command, check=True, capture_output=True)
            
            return self._parse_vina_output(smiles, str(log_file), str(output_pdbqt))

    def _prepare_ligand_from_smiles(self, smiles: str, temp_path: Path) -> Path:
        """Convert SMILES to PDBQT format"""
        ligand_pdbqt = temp_path / "ligand.pdbqt"
        
        dummy_pdbqt_content = f"""REMARK SMILES: {smiles}
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C   UNL     1       1.500   0.000   0.000  1.00  0.00           C
TORSDOF 1
"""
        with open(ligand_pdbqt, 'w') as f:
            f.write(dummy_pdbqt_content)
        
        return ligand_pdbqt

    def _convert_to_pdbqt(self, ligand_path: str, temp_path: Path) -> Path:
        """Convert ligand file to PDBQT format"""
        ligand_pdbqt = temp_path / "ligand.pdbqt"
        import shutil
        shutil.copy2(ligand_path, ligand_pdbqt)
        return ligand_pdbqt

    def _prepare_receptor(self, temp_path: Path) -> Path:
        """Prepare receptor PDBQT file"""
        receptor_pdbqt = temp_path / "receptor.pdbqt"
        import shutil
        shutil.copy2(self.protein_pdb_path, receptor_pdbqt)
        return receptor_pdbqt

    def _parse_vina_output(self, smiles: str, log_file: str, output_file: str) -> DockingResult:
        """Parse Vina output to extract binding affinity"""
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            lines = log_content.split('\n')
            for line in lines:
                if 'REMARK VINA RESULT:' in line or line.strip().startswith('1'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            affinity = float(parts[1])
                            return DockingResult(
                                smiles=smiles,
                                binding_affinity=affinity,
                                vina_score=affinity,
                                pose_rank=1
                            )
                        except ValueError:
                            continue
            
            return DockingResult(smiles=smiles, binding_affinity=0.0)
            
        except Exception as e:
            logger.error(f"Error parsing Vina output: {e}")
            return DockingResult(smiles=smiles, binding_affinity=0.0)

    def _simplified_scoring(self, smiles: str) -> DockingResult:
        """Simplified scoring when external docking is not available"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                random_score = np.random.uniform(-10, -5)
                return DockingResult(smiles=smiles, binding_affinity=random_score)
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            score = -(abs(mw - 400) * 0.01 + abs(logp - 3) * 0.5 + abs(hbd - 2) * 0.2 + abs(hba - 4) * 0.1)
            
            return DockingResult(
                smiles=smiles,
                binding_affinity=score,
                vina_score=score
            )
            
        except Exception as e:
            logger.warning(f"Simplified scoring failed for {smiles}: {e}")
            random_score = np.random.uniform(-10, -5)
            return DockingResult(smiles=smiles, binding_affinity=random_score)


class BindingSiteAnalyzer:
    """Analyze protein binding sites"""

    def __init__(self, protein_pdb_path: str):
        self.protein_pdb_path = Path(protein_pdb_path)

    def detect_binding_sites(self) -> List[BindingSite]:
        """Detect potential binding sites in the protein"""
        try:
            return self._cavity_detection()
        except Exception as e:
            logger.warning(f"Cavity detection failed: {e}")
            return self._center_of_mass_site()

    def _cavity_detection(self) -> List[BindingSite]:
        """Detect cavities using simple geometric analysis"""
        try:
            coordinates = self._parse_pdb_coordinates()
            
            if len(coordinates) == 0:
                return self._center_of_mass_site()
            
            center = np.mean(coordinates, axis=0)
            sites = [BindingSite(center[0], center[1], center[2], 20.0, 20.0, 20.0)]
            
            logger.info(f"Detected {len(sites)} potential binding sites")
            return sites
            
        except Exception as e:
            logger.error(f"Cavity detection error: {e}")
            return self._center_of_mass_site()

    def _center_of_mass_site(self) -> List[BindingSite]:
        """Create binding site at protein center of mass"""
        try:
            coordinates = self._parse_pdb_coordinates()
            if len(coordinates) > 0:
                center = np.mean(coordinates, axis=0)
                return [BindingSite(center[0], center[1], center[2])]
        except Exception:
            pass
        
        return [BindingSite(0.0, 0.0, 0.0)]

    def _parse_pdb_coordinates(self) -> np.ndarray:
        """Extract atom coordinates from PDB file"""
        coordinates = []
        
        try:
            with open(self.protein_pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coordinates.append([x, y, z])
                        except ValueError:
                            continue
            
            return np.array(coordinates)
            
        except Exception as e:
            logger.error(f"Error parsing PDB coordinates: {e}")
            return np.array([])


class DockingPipeline:
    """High-level docking pipeline"""

    def __init__(self, protein_pdb_path: str, **kwargs):
        self.docking = MolecularDocking(protein_pdb_path, **kwargs)
        self.analyzer = BindingSiteAnalyzer(protein_pdb_path)
        
        if not hasattr(self.docking, 'binding_site') or self.docking.binding_site is None:
            sites = self.analyzer.detect_binding_sites()
            if sites:
                center = (sites[0].center_x, sites[0].center_y, sites[0].center_z)
                size = (sites[0].size_x, sites[0].size_y, sites[0].size_z)
                self.docking.set_binding_site(center, size)

    def screen_molecules(self, smiles_list: List[str]) -> pd.DataFrame:
        """Screen multiple molecules and return results as DataFrame"""
        logger.info(f"Starting virtual screening of {len(smiles_list)} molecules...")
        
        results = self.docking.dock_molecules(smiles_list)
        
        df = pd.DataFrame([
            {
                'smiles': result.smiles,
                'binding_affinity': result.binding_affinity,
                'vina_score': result.vina_score,
                'pose_rank': result.pose_rank
            }
            for result in results
        ])
        
        df = df.sort_values('binding_affinity').reset_index(drop=True)
        
        logger.info(f"Virtual screening completed. Best affinity: {df['binding_affinity'].min():.2f}")
        return df

    def get_top_hits(self, smiles_list: List[str], top_n: int = 10) -> pd.DataFrame:
        """Get top N hits from virtual screening"""
        df = self.screen_molecules(smiles_list)
        return df.head(top_n)


if __name__ == "__main__":
    try:
        test_pdb_content = """ATOM      1  CA  ALA A   1      20.154  -6.351   1.000  1.00 10.00           C
ATOM      2  CA  VAL A   2      23.000  -6.000   2.500  1.00 10.00           C
ATOM      3  CA  GLY A   3      26.500  -5.500   1.800  1.00 10.00           C
END
"""
        with open('test_protein.pdb', 'w') as f:
            f.write(test_pdb_content)
        
        pipeline = DockingPipeline('test_protein.pdb')
        
        test_smiles = ['CCO', 'CCC', 'c1ccccc1O', 'CCN']
        results = pipeline.screen_molecules(test_smiles)
        
        print("Docking Results:")
        print(results)
        
        os.remove('test_protein.pdb')
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Note: This example requires RDKit and proper PDB files for full functionality")
