#!/usr/bin/env python3
"""
Chemical Space Analytics Module
Provides advanced chemical space visualization and property correlation analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple, Optional


class ChemicalSpaceAnalyzer:
    """Analyzes chemical space and molecular property relationships"""

    def __init__(self):
        self.molecules_data = []
        self.fingerprints = []
        self.properties = {}
        self.pca_model = None
        self.tsne_model = None
        self.scaler = StandardScaler()
        self.example_molecules = self.load_example_dataset()

    def load_example_dataset(self) -> List[Dict]:
        """Load example molecules for chemical space analysis"""
        examples = [
            {"name": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "activity": "anti-inflammatory"},
            {"name": "ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "activity": "anti-inflammatory"},
            {"name": "paracetamol", "smiles": "CC(=O)NC1=CC=C(C=C1)O", "activity": "analgesic"},
            {"name": "caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "activity": "stimulant"},
            {"name": "metformin", "smiles": "CN(C)C(=N)N=C(N)N", "activity": "antidiabetic"},
            {"name": "omeprazole", "smiles": "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1", "activity": "proton_pump_inhibitor"},
            {"name": "metoprolol", "smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "activity": "beta_blocker"},
            {"name": "losartan", "smiles": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(c2ccccc2c2nnn[nH]2)cc1", "activity": "arb"},
        ]
        return examples

    def calculate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate comprehensive molecular properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        try:
            properties = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "formal_charge": Chem.rdmolops.GetFormalCharge(mol),
                "ring_count": Descriptors.RingCount(mol),
                "qed": Descriptors.qed(mol),
                "lipinski_violations": self.count_lipinski_violations(mol),
            }
            return properties
        except Exception as e:
            print(f"Error calculating properties for {smiles}: {e}")
            return {}

    def count_lipinski_violations(self, mol) -> int:
        """Count Lipinski Rule of 5 violations"""
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        return violations

    def generate_fingerprint(self, smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
        """Generate Morgan fingerprint for molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits)

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            array = np.zeros((nBits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, array)
            return array
        except:
            return np.zeros(nBits)

    def prepare_dataset(self, molecules: List[Dict], include_examples: bool = True) -> pd.DataFrame:
        """Prepare dataset with molecules and their properties"""
        all_molecules = molecules.copy()
        if include_examples:
            all_molecules.extend(self.example_molecules)

        data = []
        fingerprints = []

        for mol_data in all_molecules:
            smiles = mol_data.get('smiles', '')
            if not smiles:
                continue

            properties = self.calculate_molecular_properties(smiles)
            if not properties:
                continue

            fp = self.generate_fingerprint(smiles)
            fingerprints.append(fp)

            row = {
                'name': mol_data.get('name', 'Unknown'),
                'smiles': smiles,
                'activity': mol_data.get('activity', 'unknown'),
                **properties
            }
            data.append(row)

        df = pd.DataFrame(data)
        self.fingerprints = np.array(fingerprints) if fingerprints else np.array([])
        return df

    def perform_pca(self, df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, float]:
        """Perform PCA on molecular fingerprints"""
        if len(self.fingerprints) == 0:
            return np.array([]), 0.0

        try:
            self.pca_model = PCA(n_components=n_components, random_state=42)
            pca_coords = self.pca_model.fit_transform(self.fingerprints)
            explained_variance = sum(self.pca_model.explained_variance_ratio_)
            return pca_coords, explained_variance
        except Exception as e:
            print(f"PCA failed: {e}")
            return np.array([]), 0.0

    def perform_tsne(self, df: pd.DataFrame, perplexity: int = 30) -> np.ndarray:
        """Perform t-SNE on molecular fingerprints"""
        if len(self.fingerprints) == 0:
            return np.array([])

        try:
            n_samples = len(self.fingerprints)
            perplexity = min(perplexity, max(5, n_samples // 3))
            self.tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
            tsne_coords = self.tsne_model.fit_transform(self.fingerprints)
            return tsne_coords
        except Exception as e:
            print(f"t-SNE failed: {e}")
            return np.array([])

    def cluster_molecules(self, coords: np.ndarray, method: str = 'kmeans', n_clusters: int = 5) -> np.ndarray:
        """Cluster molecules in reduced space"""
        if len(coords) == 0:
            return np.array([])

        try:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=min(n_clusters, len(coords)), random_state=42)
            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=2)
            else:
                return np.zeros(len(coords))

            clusters = clusterer.fit_predict(coords)
            return clusters
        except Exception as e:
            print(f"Clustering failed: {e}")
            return np.zeros(len(coords))

    def create_chemical_space_plot(self, df: pd.DataFrame, coords: np.ndarray, 
                                   method: str = "PCA", color_by: str = "activity") -> go.Figure:
        """Create interactive chemical space visualization"""
        if len(coords) == 0 or len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available for visualization", showarrow=False, x=0.5, y=0.5)
            return fig

        hover_text = []
        for _, row in df.iterrows():
            text = f"<b>{row['name']}</b><br>"
            text += f"MW: {row.get('molecular_weight', 0):.1f}<br>"
            text += f"LogP: {row.get('logp', 0):.2f}<br>"
            text += f"Activity: {row.get('activity', 'Unknown')}"
            hover_text.append(text)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers',
            marker=dict(size=10, color=list(range(len(coords))), colorscale='Viridis', opacity=0.7),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
        ))

        fig.update_layout(
            title=f'Chemical Space Visualization ({method})',
            xaxis_title=f'{method} Component 1',
            yaxis_title=f'{method} Component 2',
            height=600,
            hovermode='closest'
        )
        return fig

    def create_property_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create property correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient numerical data", showarrow=False, x=0.5, y=0.5)
            return fig

        corr_matrix = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0
        ))
        fig.update_layout(title='Molecular Property Correlations', height=600)
        return fig


def render_chemical_space_analytics(analyzer=None, target_smiles=None, target_name=None):
    """Render the chemical space analytics interface in Streamlit"""
    if analyzer is None:
        analyzer = ChemicalSpaceAnalyzer()

    st.info("🔬 Analyzing chemical space with molecular fingerprints...")

    # Prepare molecules
    molecules_data = []
    if target_smiles and target_name:
        molecules_data.append({"name": target_name, "smiles": target_smiles, "activity": "target"})

    # Prepare dataset
    df = analyzer.prepare_dataset(molecules_data, include_examples=True)

    if len(df) == 0:
        st.warning("No valid molecules found for analysis")
        return

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Molecules", len(df))
    with col2:
        st.metric("Unique Activities", df['activity'].nunique())
    with col3:
        drug_like = len(df[df['qed'] > 0.5]) if 'qed' in df.columns else 0
        st.metric("Drug-like (QED>0.5)", drug_like)
    with col4:
        lipinski_ok = len(df[df['lipinski_violations'] == 0]) if 'lipinski_violations' in df.columns else 0
        st.metric("Lipinski Compliant", lipinski_ok)

    # Dimensionality reduction
    st.subheader("📊 Chemical Space Visualization")
    method = st.radio("Dimensionality Reduction Method:", ["PCA", "t-SNE"], horizontal=True)

    if method == "PCA":
        coords, variance = analyzer.perform_pca(df)
        if variance > 0:
            st.info(f"PCA explained variance: {variance:.1%}")
    else:
        coords = analyzer.perform_tsne(df)

    if len(coords) > 0:
        fig = analyzer.create_chemical_space_plot(df, coords, method=method)
        st.plotly_chart(fig, use_container_width=True)

        # Highlight target molecule
        if target_name:
            target_idx = df[df['name'] == target_name].index
            if len(target_idx) > 0:
                st.success(f"🎯 Your molecule '{target_name}' is highlighted in the chemical space")

    # Property correlation
    st.subheader("📈 Property Correlation Matrix")
    corr_fig = analyzer.create_property_correlation_matrix(df)
    st.plotly_chart(corr_fig, use_container_width=True)

    # Show data table
    st.subheader("📋 Molecule Data")
    display_cols = ['name', 'activity', 'molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'qed', 'lipinski_violations']
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols].round(2), use_container_width=True)
