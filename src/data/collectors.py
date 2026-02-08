"""
Data Collection Module for Drug Discovery Pipeline

Interfaces with major chemical and biological databases:
    - ChEMBL: Bioactive molecules + activity data
    - PubChem: Chemical structures + biological assays
    - ZINC: Ready-to-dock 3D compounds
    - BindingDB: Protein-ligand binding affinities
    - RCSB PDB: Protein 3D structures
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MoleculeData:
    """Data structure for molecule information"""
    smiles: str
    molecule_id: str
    activity_value: Optional[float] = None
    activity_type: Optional[str] = None
    target_id: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ProteinData:
    """Data structure for protein information"""
    pdb_id: str
    sequence: str
    structure_path: Optional[str] = None
    binding_sites: Optional[List[Dict]] = None
    resolution: Optional[float] = None


class DataCollector(ABC):
    """Abstract base class for data collectors"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def search(self, query: str, limit: int = 1000) -> List[MoleculeData]:
        """Search for molecules matching query"""
        pass

    @abstractmethod
    def get_by_id(self, molecule_id: str) -> Optional[MoleculeData]:
        """Get molecule by ID"""
        pass


class ChEMBLCollector(DataCollector):
    """Collect data from ChEMBL database"""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"

    def search_target(self, target_name: str) -> Dict:
        """Search for protein targets"""
        url = f"{self.base_url}/target"
        params = {
            'pref_name__icontains': target_name,
            'format': 'json'
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"ChEMBL target search failed: {response.status_code}")
            return {}

    def get_target_activities(self, target_chembl_id: str, limit: int = 1000) -> List[MoleculeData]:
        """Get bioactive molecules for a specific target"""
        url = f"{self.base_url}/activity"
        params = {
            'target_chembl_id': target_chembl_id,
            'standard_type__in': 'IC50,Ki,Kd,EC50',
            'standard_value__isnull': 'false',
            'limit': limit,
            'format': 'json'
        }

        molecules = []
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for activity in data.get('activities', []):
                    if activity.get('canonical_smiles'):
                        molecule = MoleculeData(
                            smiles=activity['canonical_smiles'],
                            molecule_id=activity.get('molecule_chembl_id', ''),
                            activity_value=float(activity.get('standard_value', 0)),
                            activity_type=activity.get('standard_type', ''),
                            target_id=target_chembl_id,
                            source='ChEMBL'
                        )
                        molecules.append(molecule)

                logger.info(f"Retrieved {len(molecules)} molecules from ChEMBL")
                return molecules
            else:
                logger.error(f"ChEMBL activity search failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching ChEMBL data: {e}")

        return []

    def search(self, query: str, limit: int = 1000) -> List[MoleculeData]:
        """Search ChEMBL by molecule name or SMILES"""
        # First try to find target, then get activities
        target_data = self.search_target(query)
        if target_data.get('targets'):
            target_id = target_data['targets'][0].get('target_chembl_id')
            if target_id:
                return self.get_target_activities(target_id, limit)
        return []

    def get_by_id(self, molecule_id: str) -> Optional[MoleculeData]:
        """Get molecule by ChEMBL ID"""
        url = f"{self.base_url}/molecule/{molecule_id}"
        params = {'format': 'json'}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return MoleculeData(
                    smiles=data.get('molecule_structures', {}).get('canonical_smiles', ''),
                    molecule_id=molecule_id,
                    source='ChEMBL'
                )
        except Exception as e:
            logger.error(f"Error fetching ChEMBL molecule {molecule_id}: {e}")

        return None


class PubChemCollector(DataCollector):
    """Collect data from PubChem database"""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def search(self, query: str, limit: int = 1000) -> List[MoleculeData]:
        """Search PubChem by name or SMILES"""
        # Search by name first
        search_url = f"{self.base_url}/compound/name/{query}/cids/JSON"

        try:
            response = requests.get(search_url)
            if response.status_code == 200:
                cids = response.json().get('IdentifierList', {}).get('CID', [])
                # Limit results
                cids = cids[:limit]

                molecules = []
                for cid in cids:
                    molecule = self.get_by_id(str(cid))
                    if molecule:
                        molecules.append(molecule)

                logger.info(f"Retrieved {len(molecules)} molecules from PubChem")
                return molecules

        except Exception as e:
            logger.error(f"Error searching PubChem: {e}")

        return []

    def get_by_id(self, cid: str) -> Optional[MoleculeData]:
        """Get molecule by PubChem CID"""
        props_url = f"{self.base_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"

        try:
            response = requests.get(props_url)
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    smiles = props[0].get('CanonicalSMILES', '')
                    return MoleculeData(
                        smiles=smiles,
                        molecule_id=cid,
                        source='PubChem'
                    )
        except Exception as e:
            logger.error(f"Error fetching PubChem compound {cid}: {e}")

        return None


class ZINCCollector(DataCollector):
    """Collect data from ZINC database"""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        self.base_url = "https://zinc.docking.org"

    def search(self, query: str, limit: int = 1000) -> List[MoleculeData]:
        """Search ZINC database - placeholder implementation"""
        # Note: ZINC doesn't have a simple REST API
        # In practice, you would download ZINC subsets or use their web interface
        logger.info("ZINC search requires manual download of subsets")
        return []

    def get_by_id(self, zinc_id: str) -> Optional[MoleculeData]:
        """Get molecule by ZINC ID - placeholder implementation"""
        logger.info("ZINC ID lookup requires manual implementation")
        return None


class BindingDBCollector(DataCollector):
    """Collect data from BindingDB"""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        self.base_url = "https://www.bindingdb.org/axis2/services/BDBService"

    def search(self, query: str, limit: int = 1000) -> List[MoleculeData]:
        """Search BindingDB - placeholder implementation"""
        # BindingDB has a web service but requires specific SOAP calls
        logger.info("BindingDB search requires SOAP interface implementation")
        return []

    def get_by_id(self, binding_id: str) -> Optional[MoleculeData]:
        """Get binding data by ID - placeholder implementation"""
        logger.info("BindingDB lookup requires SOAP interface implementation")
        return None


class PDBCollector:
    """Collect protein structure data from RCSB PDB"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://data.rcsb.org/rest/v1/core"

    def get_structure_info(self, pdb_id: str) -> Dict:
        """Get structure information"""
        url = f"{self.base_url}/entry/{pdb_id.upper()}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching PDB info for {pdb_id}: {e}")

        return {}

    def download_structure(self, pdb_id: str, format: str = 'pdb') -> Optional[str]:
        """Download PDB structure file"""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{format}"
        output_path = self.cache_dir / f"{pdb_id.upper()}.{format}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, 'w') as f:
                    f.write(response.text)
                logger.info(f"Downloaded PDB structure: {output_path}")
                return str(output_path)
        except Exception as e:
            logger.error(f"Error downloading PDB {pdb_id}: {e}")

        return None


class DataManager:
    """Main data management interface"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.chembl = ChEMBLCollector(cache_dir)
        self.pubchem = PubChemCollector(cache_dir)
        self.zinc = ZINCCollector(cache_dir)
        self.bindingdb = BindingDBCollector(cache_dir)
        self.pdb = PDBCollector(cache_dir)

    def collect_target_data(self, target_name: str, limit: int = 1000) -> pd.DataFrame:
        """Collect comprehensive data for a target"""
        all_molecules = []

        # Collect from ChEMBL
        chembl_data = self.chembl.search(target_name, limit)
        all_molecules.extend(chembl_data)

        # Collect from PubChem
        pubchem_data = self.pubchem.search(target_name, limit)
        all_molecules.extend(pubchem_data)

        # Convert to DataFrame
        if all_molecules:
            df = pd.DataFrame([
                {
                    'smiles': mol.smiles,
                    'molecule_id': mol.molecule_id,
                    'activity_value': mol.activity_value,
                    'activity_type': mol.activity_type,
                    'target_id': mol.target_id,
                    'source': mol.source
                }
                for mol in all_molecules
            ])

            # Remove duplicates based on SMILES
            df = df.drop_duplicates(subset=['smiles'])
            logger.info(f"Collected {len(df)} unique molecules for target '{target_name}'")
            return df

        return pd.DataFrame()

    def get_protein_structure(self, pdb_id: str) -> Optional[ProteinData]:
        """Get protein structure and metadata"""
        info = self.pdb.get_structure_info(pdb_id)
        structure_path = self.pdb.download_structure(pdb_id)

        if info and structure_path:
            return ProteinData(
                pdb_id=pdb_id.upper(),
                sequence='',  # Would extract from structure file
                structure_path=structure_path,
                resolution=info.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]
            )

        return None


# Example usage
if __name__ == "__main__":
    manager = DataManager()

    # Collect data for a specific target
    target_data = manager.collect_target_data("EGFR", limit=100)
    print(f"Collected {len(target_data)} molecules for EGFR")

    # Get protein structure
    protein = manager.get_protein_structure("1M17")  # EGFR structure
    if protein:
        print(f"Downloaded structure: {protein.structure_path}")
