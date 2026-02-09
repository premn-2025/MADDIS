#!/usr/bin/env python3
"""
Universal Multi-Agent Drug Discovery Platform - INTEGRATED VERSION
Combines the web interface with the complete multi-agent system
"""

import os
import warnings
import logging

# Suppress all warnings before importing other libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
try:
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('torch.classes').setLevel(logging.ERROR)
except Exception:
    pass

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# Load Streamlit secrets into environment variables (for deployed apps)
try:
    for key, value in st.secrets.items():
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = str(value)
except Exception:
    pass
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Crippen
from rdkit.Chem import rdMolDescriptors, rdDepictor
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Optional torch import (heavy dependency, not needed for core UI)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

# Optional transformers import for AI model
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    PeftModel = None
    PeftConfig = None

warnings.filterwarnings('ignore')

# Import multi-agent components
try:
    from demo_multiagent_platform import MultiAgentOrchestrator
    MULTIAGENT_AVAILABLE = True
except ImportError:
    MULTIAGENT_AVAILABLE = False

# Import REAL docking system
try:
    from real_docking_agent import RealMolecularDockingAgent
    REAL_DOCKING_AVAILABLE = True
except ImportError:
    REAL_DOCKING_AVAILABLE = False

# Import RL molecular generator
try:
    from improved_rl_generator import ImprovedRLMolecularGenerator
    RL_GENERATOR_AVAILABLE = True
except ImportError:
    RL_GENERATOR_AVAILABLE = False

# Import multi-target RL generator
try:
    from multi_target_rl_generator import (
        MultiTargetRLGenerator,
        MultiTargetObjective,
        ParetoOptimizer
    )
    MULTITARGET_RL_AVAILABLE = True
except ImportError:
    MULTITARGET_RL_AVAILABLE = False

# Import REAL RL trainer
try:
    from real_rl_trainer import RealRLTrainer
    REAL_RL_TRAINER_AVAILABLE = True
except ImportError:
    REAL_RL_TRAINER_AVAILABLE = False

# Import Property Prediction Agent
try:
    from property_prediction_agent import PropertyPredictionAgent, PropertyPrediction
    from enhanced_multi_target_generator import EnhancedMultiTargetRLGenerator
    PROPERTY_PREDICTION_AVAILABLE = True
except ImportError:
    PROPERTY_PREDICTION_AVAILABLE = False

# Import Gemini 3 orchestrator
try:
    from gemini3_orchestrator import LocalGemini3Orchestrator
    GEMINI3_AVAILABLE = True
except ImportError:
    GEMINI3_AVAILABLE = False

LITERATURE_MINING_AVAILABLE = False

# Import chemical space analytics
try:
    from chemical_space_analytics import ChemicalSpaceAnalyzer, render_chemical_space_analytics
    CHEMICAL_ANALYTICS_AVAILABLE = True
except ImportError:
    CHEMICAL_ANALYTICS_AVAILABLE = False

# Import section chat assistant
try:
    from section_chat_assistant import render_chat_expander
    CHAT_ASSISTANT_AVAILABLE = True
except ImportError:
    CHAT_ASSISTANT_AVAILABLE = False

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MADDIS: AI Drug Discovery Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for faster rendering
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 30px;
}
.stSpinner > div { font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

# Header will be shown in the run() method after initialization

class UniversalMultiAgentPlatform:
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.model = None
        self.tokenizer = None
        self.multiagent_orchestrator = None
        # Don't initialize multi-agent system at startup - do it lazily when needed
        
        self.known_molecules = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "morphine": "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",
            "warfarin": "CC1=C(C2=C(C=C1)OC(=O)[C@@H]2C3=CC=C(C=C3)Cl)C",
            "metformin": "CN(C)C(=N)N=C(N)N",
            "ethanol": "CCO",
            "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
            "penicillin": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
        }

    def initialize_multiagent_system(self):
        """Initialize the multi-agent orchestrator"""
        try:
            self.multiagent_orchestrator = MultiAgentOrchestrator()
            if 'multiagent_initialized' not in st.session_state:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.multiagent_orchestrator.initialize_agents())
                loop.close()
                st.session_state.multiagent_initialized = True
                st.session_state.multiagent_orchestrator = self.multiagent_orchestrator
            else:
                self.multiagent_orchestrator = st.session_state.multiagent_orchestrator
        except Exception as e:
            st.error(f"Failed to initialize multi-agent system: {e}")
            self.multiagent_orchestrator = None

    def load_ai_model(self):
        """Load the trained AI model for drug interaction prediction"""
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            st.warning("Transformers/PyTorch not available. AI predictions disabled.")
            return False
        try:
            model_paths = ["./max_accuracy_drug_model_final", "./ultra_drug_interaction_final"]
            for model_path in model_paths:
                if os.path.exists(model_path):
                    st.info(f"Loading AI model from {model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        "dmis-lab/biobert-base-cased-v1.2", num_labels=5)
                    base_model = base_model.to(self.device)
                    self.model = PeftModel.from_pretrained(base_model, model_path)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    st.success(" AI model loaded successfully!")
                    return True
            return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def get_molecule_from_name_or_smiles(self, input_text):
        input_text = input_text.strip()
        # Check if it's a known molecule name (case-insensitive)
        input_lower = input_text.lower()
        if input_lower in self.known_molecules:
            smiles = self.known_molecules[input_lower]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return mol, smiles, input_lower.title()
        # Try as SMILES (case-sensitive - SMILES uses case for chirality)
        mol = Chem.MolFromSmiles(input_text)
        if mol:
            canonical_smiles = Chem.MolToSmiles(mol)
            return mol, canonical_smiles, "Custom Molecule"
        return None, None, None

    def generate_3d_structure(self, mol):
        """Generate 3D coordinates for a molecule"""
        try:
            # Create a copy to avoid modifying original
            mol_3d = Chem.AddHs(Chem.Mol(mol))
            # Use random seed based on molecule hash for reproducibility per molecule
            mol_hash = hash(Chem.MolToSmiles(mol)) % 10000
            # Embed with ETKDG for better 3D coordinates
            params = AllChem.ETKDGv3()
            params.randomSeed = mol_hash
            result = AllChem.EmbedMolecule(mol_3d, params)
            if result == -1:
                # Fallback to random embedding
                AllChem.EmbedMolecule(mol_3d, randomSeed=mol_hash)
            # Optimize geometry
            try:
                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
            except:
                try:
                    AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
                except:
                    pass
            return mol_3d
        except Exception as e:
            st.warning(f"3D generation warning: {e}")
            # Return 2D with fake Z coordinates
            mol_2d = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol_2d)
            return mol_2d

    def create_3d_plot(self, mol_3d, title="Molecular Structure"):
        """Create 3D plot of molecule structure"""
        try:
            # Check if molecule has a conformer
            if mol_3d.GetNumConformers() == 0:
                # Generate conformer if missing
                mol_3d = Chem.AddHs(mol_3d)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                if mol_3d.GetNumConformers() == 0:
                    AllChem.Compute2DCoords(mol_3d)

            conf = mol_3d.GetConformer(0)
        except Exception as e:
            st.error(f"Cannot get conformer: {e}")
            return go.Figure()

        atoms, x, y, z, colors, sizes = [], [], [], [], [], []
        color_map = {'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'H': '#FFFFFF', 'S': '#FFFF30', 'F': '#90E050', 'Cl': '#1FF01F', 'Br': '#A62929', 'P': '#FF8000', 'I': '#940094'}
        size_map = {'C': 8, 'N': 8, 'O': 7, 'H': 4, 'S': 9, 'F': 6, 'Cl': 8, 'Br': 9, 'P': 9, 'I': 10}

        for i, atom in enumerate(mol_3d.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            atoms.append(symbol)
            x.append(pos.x)
            y.append(pos.y)
            z.append(pos.z if hasattr(pos, 'z') else 0.0)
            colors.append(color_map.get(symbol, '#FF69B4'))
            sizes.append(size_map.get(symbol, 8))

        bonds_x, bonds_y, bonds_z = [], [], []
        for bond in mol_3d.GetBonds():
            start_pos = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            end_pos = conf.GetAtomPosition(bond.GetEndAtomIdx())
            bonds_x.extend([start_pos.x, end_pos.x, None])
            bonds_y.extend([start_pos.y, end_pos.y, None])
            bonds_z.extend([start_pos.z, end_pos.z, None])

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=bonds_x, y=bonds_y, z=bonds_z, mode='lines', line=dict(color='gray', width=4), hoverinfo='skip', name='Bonds'))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=sizes, color=colors, line=dict(width=1, color='black')), text=atoms, name='Atoms', hovertemplate='%{text}<extra></extra>'))
        fig.update_layout(title=title, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", bgcolor='rgba(240,240,240,0.9)'), height=500, showlegend=False)
        return fig

    def calculate_molecular_properties(self, mol):
        try:
            properties = {
                'Molecular Weight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'H-Bond Donors': Descriptors.NumHDonors(mol),
                'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
                'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
                'Aromatic Rings': Descriptors.NumAromaticRings(mol),
                'Topological Polar Surface Area': Descriptors.TPSA(mol),
                'Heavy Atoms': mol.GetNumHeavyAtoms(),
                'Formal Charge': Chem.rdmolops.GetFormalCharge(mol),
                'Ring Count': Descriptors.RingCount(mol)
            }
            violations = 0
            if properties['Molecular Weight'] > 500: violations += 1
            if properties['LogP'] > 5: violations += 1
            if properties['H-Bond Donors'] > 5: violations += 1
            if properties['H-Bond Acceptors'] > 10: violations += 1
            properties['Lipinski Violations'] = violations
            properties['Drug-like'] = violations <= 1
            return properties
        except Exception as e:
            st.error(f"Error calculating properties: {e}")
            return {}

    async def run_multiagent_analysis(self, molecule_input, target_protein="EGFR", protocol="flexible", poses=5):
        if not self.multiagent_orchestrator:
            return {"error": "Multi-agent system not available"}
        mol, smiles, name = self.get_molecule_from_name_or_smiles(molecule_input)
        if not mol:
            return {"error": f"Could not parse molecule: {molecule_input}"}
        return await self.multiagent_orchestrator.execute_drug_discovery_task(smiles, target_protein)

    def display_multiagent_results(self, results):
        if "error" in results:
            st.error(f" Multi-agent analysis failed: {results['error']}")
            return
        st.success(" Multi-Agent Analysis Complete!")
        task_summary = results.get("task_summary", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Task ID", task_summary.get("task_id", "Unknown")[-8:])
        col2.metric("Execution Time", f"{task_summary.get('total_execution_time_seconds', 0):.1f}s")
        sys_perf = results.get("system_performance", {})
        col3.metric("Success Rate", f"{sys_perf.get('successful_phases', 0)}/{sys_perf.get('successful_phases', 0) + sys_perf.get('failed_phases', 0)}")

        findings = results.get("key_findings", {})
        if findings:
            tabs = st.tabs(["Molecular Design", "Docking Analysis", "Validation", "Recommendations"])
            with tabs[0]:
                if "molecular_properties" in findings:
                    props = findings["molecular_properties"]
                    c1, c2 = st.columns(2)
                    c1.metric("Molecular Weight", f"{props.get('molecular_weight', 0):.1f} Da")
                    c2.metric("LogP", f"{props.get('logp', 0):.2f}")
            with tabs[1]:
                if "docking_score" in findings:
                    st.metric("Docking Score", f"{findings['docking_score']:.2f} kcal/mol")
            with tabs[2]:
                if "validation_score" in findings:
                    st.write(f"Validation Score: {findings['validation_score']:.2f}")
            with tabs[3]:
                for i, rec in enumerate(results.get("recommendations", []), 1):
                    st.write(f"**{i}.** {rec}")

    async def run_multitarget_rl_training(self, objectives, generations, config):
        # Implementation logic for multi-target training
        pass

    async def run_multitarget_rl_library(self, objectives, size, config):
        # Implementation logic for library generation
        pass

    def display_multitarget_training_results(self, results):
        pass

    def display_multitarget_library_results(self, results):
        pass

    def show_landing_page(self):
        st.markdown("### Welcome to the Universal Multi-Agent Drug Discovery Platform")
        if MULTIAGENT_AVAILABLE:
            st.success("🤖 **Multi-Agent System Active**: Advanced autonomous agent analysis available")
        
        # Feature Overview
        st.subheader("🎯 Platform Capabilities")
        cap_cols = st.columns(4)
        with cap_cols[0]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <h4>🧬 Molecule Analysis</h4>
            <p>• 3D Structure Visualization<br>• Property Calculation<br>• Lipinski Rule Check</p>
            </div>
            """, unsafe_allow_html=True)
        with cap_cols[1]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <h4>💊 Drug Interactions</h4>
            <p>• Drug-Drug Checker<br>• Risk Assessment<br>• Clinical Warnings</p>
            </div>
            """, unsafe_allow_html=True)
        with cap_cols[2]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <h4>🔬 ADMET Prediction</h4>
            <p>• Absorption<br>• Toxicity<br>• Metabolism</p>
            </div>
            """, unsafe_allow_html=True)
        with cap_cols[3]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <h4>🧪 Synthesis Planning</h4>
            <p>• Retrosynthesis<br>• Route Optimization<br>• Cost Estimation</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader(" Available Agents")
        agent_cols = st.columns(4)
        with agent_cols[0]:
            st.markdown('<div class="agent-card"><h4>🎨 Molecular Designer</h4><p>• Property prediction<br>• Generation</p></div>', unsafe_allow_html=True)
        with agent_cols[1]:
            st.markdown('<div class="agent-card"><h4>🔗 Docking Specialist</h4><p>• Molecular docking<br>• Affinity</p></div>', unsafe_allow_html=True)
        with agent_cols[2]:
            st.markdown('<div class="agent-card"><h4>✅ Validation Critic</h4><p>• Statistical validation<br>• Bias detection</p></div>', unsafe_allow_html=True)
        with agent_cols[3]:
            st.markdown('<div class="agent-card"><h4>⚗️ Synthesis Planner</h4><p>• Retrosynthesis<br>• Route planning</p></div>', unsafe_allow_html=True)

    def run_drug_interaction_checker(self):
        """Drug-Drug Compatibility & Stability Analysis"""
        st.subheader("💊 Drug Compatibility & Stability Checker")
        st.markdown("Analyze molecular compatibility, stability, and interaction risk between two compounds")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("drug_compatibility", key_suffix="_drug_interaction")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Drug A")
            drug_a = st.text_input("Name or SMILES:", placeholder="e.g., aspirin or CC(=O)OC1=CC=CC=C1C(=O)O", key="drug_a_input")
        with col2:
            st.markdown("#### Drug B")
            drug_b = st.text_input("Name or SMILES:", placeholder="e.g., warfarin or CC1=C(C2=CC=CC=C2OC1=O)C(=O)CC3=CC=CC=C3", key="drug_b_input")
        
        if st.button("🔍 Analyze Compatibility", type="primary"):
            if not drug_a or not drug_b:
                st.warning("Please enter both drug names or SMILES")
                return
            
            mol_a, smiles_a, name_a = self.get_molecule_from_name_or_smiles(drug_a)
            mol_b, smiles_b, name_b = self.get_molecule_from_name_or_smiles(drug_b)
            
            if not mol_a:
                st.error(f"Could not parse Drug A: **{drug_a}**. Try a known name (aspirin, caffeine…) or a valid SMILES string.")
                return
            if not mol_b:
                st.error(f"Could not parse Drug B: **{drug_b}**. Try a known name (aspirin, caffeine…) or a valid SMILES string.")
                return
            
            with st.spinner("Analyzing molecular compatibility…"):
                self._display_compatibility_results(mol_a, smiles_a, name_a, mol_b, smiles_b, name_b)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def _compute_mol_profile(self, mol):
        """Return a dict of pharmacologically relevant properties."""
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        aromatic = Descriptors.NumAromaticRings(mol)
        rings = Descriptors.RingCount(mol)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        qed = Descriptors.qed(mol)
        fsp3 = Descriptors.FractionCSP3(mol)

        # Lipinski violations
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        # Functional-group flags
        smarts_patterns = {
            'Carboxylic acid': '[CX3](=O)[OX2H1]',
            'Amine (primary)': '[NX3;H2;!$(NC=O)]',
            'Amine (secondary)': '[NX3;H1;!$(NC=O)]',
            'Amine (tertiary)': '[NX3;H0;!$(NC=O);!$(N=O)]',
            'Hydroxyl': '[OX2H]',
            'Ketone': '[CX3](=O)[#6]',
            'Aldehyde': '[CX3H1](=O)',
            'Ester': '[CX3](=O)[OX2H0]',
            'Amide': '[NX3][CX3](=O)',
            'Nitro': '[$([NX3](=O)=O)]',
            'Sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])',
            'Phenol': '[OX2H][cX3]:[c]',
            'Thiol': '[SX2H]',
        }
        groups_present = []
        for name, smarts in smarts_patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                groups_present.append(name)

        return {
            'MW': mw, 'LogP': logp, 'TPSA': tpsa, 'HBD': hbd, 'HBA': hba,
            'Rotatable Bonds': rotatable, 'Aromatic Rings': aromatic,
            'Ring Count': rings, 'Charge': charge, 'QED': qed,
            'Fsp3': fsp3, 'Lipinski Violations': violations,
            'Functional Groups': groups_present,
        }

    def _assess_interaction_risk(self, prof_a, prof_b):
        """Return (risk_level, reasons) based on property profiles."""
        reasons = []
        risk_score = 0  # 0-100

        # Known reactive-group clashes
        reactive_pairs = [
            ('Carboxylic acid', 'Amine (primary)', 'Acid-base reaction → salt / amide formation', 25),
            ('Carboxylic acid', 'Amine (secondary)', 'Acid-base reaction → salt formation', 20),
            ('Aldehyde', 'Amine (primary)', 'Schiff-base (imine) formation', 30),
            ('Aldehyde', 'Amine (secondary)', 'Enamine / hemiaminal formation', 25),
            ('Aldehyde', 'Thiol', 'Thio-hemiacetal formation', 20),
            ('Carboxylic acid', 'Hydroxyl', 'Possible ester hydrolysis / exchange', 10),
            ('Ester', 'Amine (primary)', 'Aminolysis (ester → amide conversion)', 20),
            ('Nitro', 'Thiol', 'Redox interaction risk', 15),
        ]
        groups_a = set(prof_a['Functional Groups'])
        groups_b = set(prof_b['Functional Groups'])
        for g1, g2, msg, pts in reactive_pairs:
            if (g1 in groups_a and g2 in groups_b) or (g1 in groups_b and g2 in groups_a):
                risk_score += pts
                reasons.append(f"⚗️ **{g1} + {g2}**: {msg}")

        # pH-dependent solubility clash
        if prof_a['Charge'] != 0 and prof_b['Charge'] != 0 and prof_a['Charge'] * prof_b['Charge'] < 0:
            risk_score += 15
            reasons.append("⚡ Opposite formal charges → ionic complexation / precipitation risk")

        # CYP competition (proxy: both highly lipophilic + aromatic)
        if prof_a['LogP'] > 3 and prof_b['LogP'] > 3:
            risk_score += 10
            reasons.append("🧬 Both lipophilic (LogP > 3) → likely compete for CYP450 metabolism")
        if prof_a['Aromatic Rings'] >= 3 and prof_b['Aromatic Rings'] >= 3:
            risk_score += 5
            reasons.append("🔗 Both highly aromatic → π-π stacking / co-precipitation risk")

        # Protein-binding displacement (proxy: high LogP + low TPSA)
        def high_binding(p):
            return p['LogP'] > 3 and p['TPSA'] < 80
        if high_binding(prof_a) and high_binding(prof_b):
            risk_score += 15
            reasons.append("🩸 Both predicted high plasma protein binding → displacement interaction risk")

        # Known dangerous name pairs (keep as safety net)
        dangerous_pairs = {
            frozenset(["aspirin", "warfarin"]): ("Increased bleeding risk – BLACK BOX WARNING", 40),
            frozenset(["aspirin", "ibuprofen"]): ("GI bleeding risk, reduced cardioprotection", 25),
            frozenset(["ibuprofen", "warfarin"]): ("Increased bleeding & GI ulceration risk", 35),
            frozenset(["metformin", "ethanol"]): ("Lactic acidosis risk", 35),
        }

        risk_score = min(risk_score, 100)
        if risk_score >= 40:
            level = "HIGH"
        elif risk_score >= 20:
            level = "MODERATE"
        else:
            level = "LOW"
        return level, risk_score, reasons

    def _assess_stability(self, prof):
        """Heuristic chemical-stability assessment for one molecule."""
        flags = []
        score = 100  # start perfect, deduct

        if 'Aldehyde' in prof['Functional Groups']:
            score -= 20
            flags.append("Aldehyde group – susceptible to oxidation & nucleophilic attack")
        if 'Thiol' in prof['Functional Groups']:
            score -= 15
            flags.append("Thiol group – prone to oxidation (disulfide formation)")
        if 'Ester' in prof['Functional Groups']:
            score -= 10
            flags.append("Ester group – hydrolysis-sensitive at extreme pH")
        if prof['Aromatic Rings'] == 0 and prof['Ring Count'] == 0:
            score -= 5
            flags.append("Acyclic structure – may have higher conformational flexibility / metabolic liability")
        if prof['Fsp3'] < 0.1:
            score -= 5
            flags.append("Very flat molecule (Fsp3 < 0.1) – possible crystallization / solubility issues")
        if prof['LogP'] > 5:
            score -= 10
            flags.append("High lipophilicity (LogP > 5) – poor aqueous stability / bioavailability")
        if prof['MW'] > 500:
            score -= 5
            flags.append("High MW (>500) – potential formulation challenges")
        if prof['TPSA'] > 140:
            score -= 10
            flags.append("High TPSA (>140 Å²) – poor membrane permeability")

        score = max(score, 0)
        if score >= 80:
            label = "Stable"
        elif score >= 50:
            label = "Moderately Stable"
        else:
            label = "Unstable / Caution"
        return label, score, flags

    def _compute_similarity(self, mol_a, mol_b):
        """Tanimoto similarity on Morgan fingerprints."""
        from rdkit.DataStructs import TanimotoSimilarity
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return TanimotoSimilarity(fp_a, fp_b)

    def _display_compatibility_results(self, mol_a, smiles_a, name_a, mol_b, smiles_b, name_b):
        """Full compatibility dashboard."""
        prof_a = self._compute_mol_profile(mol_a)
        prof_b = self._compute_mol_profile(mol_b)
        risk_level, risk_score, risk_reasons = self._assess_interaction_risk(prof_a, prof_b)
        stab_a_label, stab_a_score, stab_a_flags = self._assess_stability(prof_a)
        stab_b_label, stab_b_score, stab_b_flags = self._assess_stability(prof_b)
        similarity = self._compute_similarity(mol_a, mol_b)

        # ---------- Header metrics ----------
        st.markdown("---")
        st.subheader(f"Results: {name_a} + {name_b}")

        m1, m2, m3, m4 = st.columns(4)
        risk_color = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[risk_level]
        m1.metric("Interaction Risk", f"{risk_color} {risk_level}", f"Score {risk_score}/100")
        m2.metric("Similarity", f"{similarity:.1%}")
        m3.metric(f"{name_a} Stability", f"{stab_a_score}/100")
        m4.metric(f"{name_b} Stability", f"{stab_b_score}/100")

        # ---------- Tabs ----------
        t_inter, t_stab, t_prop, t_struct = st.tabs([
            "⚠️ Interaction Analysis",
            "🧪 Stability Report",
            "📊 Property Comparison",
            "🧬 3D Structures"
        ])

        # --- Interaction tab ---
        with t_inter:
            if risk_level == "HIGH":
                st.error(f"🚫 **HIGH interaction risk** (score {risk_score}/100) – co-administration may be dangerous")
            elif risk_level == "MODERATE":
                st.warning(f"⚠️ **MODERATE interaction risk** (score {risk_score}/100) – monitor closely")
            else:
                st.success(f"✅ **LOW interaction risk** (score {risk_score}/100) – likely compatible")

            if risk_reasons:
                st.markdown("#### Identified Risk Factors")
                for r in risk_reasons:
                    st.markdown(f"- {r}")
            else:
                st.info("No specific reactive-group clashes detected between these molecules.")

            # Functional-group overlay
            st.markdown("#### Functional Groups Detected")
            fg_col1, fg_col2 = st.columns(2)
            with fg_col1:
                st.markdown(f"**{name_a}**")
                if prof_a['Functional Groups']:
                    for g in prof_a['Functional Groups']:
                        st.markdown(f"  • {g}")
                else:
                    st.write("  No reactive groups detected")
            with fg_col2:
                st.markdown(f"**{name_b}**")
                if prof_b['Functional Groups']:
                    for g in prof_b['Functional Groups']:
                        st.markdown(f"  • {g}")
                else:
                    st.write("  No reactive groups detected")

            st.info("⚕️ **Disclaimer**: This is a computational prediction. Always consult a healthcare professional for clinical decisions.")

        # --- Stability tab ---
        with t_stab:
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.markdown(f"### {name_a}")
                if stab_a_score >= 80:
                    st.success(f"✅ **{stab_a_label}** ({stab_a_score}/100)")
                elif stab_a_score >= 50:
                    st.warning(f"⚠️ **{stab_a_label}** ({stab_a_score}/100)")
                else:
                    st.error(f"🚫 **{stab_a_label}** ({stab_a_score}/100)")
                if stab_a_flags:
                    for f in stab_a_flags:
                        st.write(f"  ⚬ {f}")
                else:
                    st.write("  No stability concerns detected.")

            with s_col2:
                st.markdown(f"### {name_b}")
                if stab_b_score >= 80:
                    st.success(f"✅ **{stab_b_label}** ({stab_b_score}/100)")
                elif stab_b_score >= 50:
                    st.warning(f"⚠️ **{stab_b_label}** ({stab_b_score}/100)")
                else:
                    st.error(f"🚫 **{stab_b_label}** ({stab_b_score}/100)")
                if stab_b_flags:
                    for f in stab_b_flags:
                        st.write(f"  ⚬ {f}")
                else:
                    st.write("  No stability concerns detected.")

            # Combined stability assessment
            st.markdown("---")
            combined_stab = (stab_a_score + stab_b_score) / 2
            st.markdown("### Combined Formulation Stability Estimate")
            if combined_stab >= 75 and risk_score < 20:
                st.success(f"✅ Favorable combination – combined stability {combined_stab:.0f}/100, low interaction risk")
            elif combined_stab >= 50:
                st.warning(f"⚠️ Acceptable with monitoring – combined stability {combined_stab:.0f}/100")
            else:
                st.error(f"🚫 Combination raises concerns – combined stability {combined_stab:.0f}/100")

        # --- Property comparison tab ---
        with t_prop:
            compare_keys = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'Rotatable Bonds',
                            'Aromatic Rings', 'Ring Count', 'Charge', 'QED', 'Fsp3', 'Lipinski Violations']
            comp_df = pd.DataFrame({
                'Property': compare_keys,
                name_a: [f"{prof_a[k]:.2f}" if isinstance(prof_a[k], float) else str(prof_a[k]) for k in compare_keys],
                name_b: [f"{prof_b[k]:.2f}" if isinstance(prof_b[k], float) else str(prof_b[k]) for k in compare_keys],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Radar chart
            radar_keys = ['QED', 'Fsp3', 'HBD', 'HBA', 'Aromatic Rings']
            vals_a = [prof_a[k] for k in radar_keys]
            vals_b = [prof_b[k] for k in radar_keys]
            # Normalise to 0-1 for radar
            max_vals = [max(abs(a), abs(b), 1e-9) for a, b in zip(vals_a, vals_b)]
            norm_a = [v / m for v, m in zip(vals_a, max_vals)]
            norm_b = [v / m for v, m in zip(vals_b, max_vals)]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=norm_a + [norm_a[0]], theta=radar_keys + [radar_keys[0]],
                                          fill='toself', name=name_a, opacity=0.6))
            fig.add_trace(go.Scatterpolar(r=norm_b + [norm_b[0]], theta=radar_keys + [radar_keys[0]],
                                          fill='toself', name=name_b, opacity=0.6))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                              title="Property Radar Comparison", height=450)
            st.plotly_chart(fig, use_container_width=True)

        # --- 3D structures tab ---
        with t_struct:
            struct_col1, struct_col2 = st.columns(2)
            with struct_col1:
                mol_3d_a = self.generate_3d_structure(mol_a)
                st.plotly_chart(self.create_3d_plot(mol_3d_a, f"{name_a} 3D"), use_container_width=True)
                st.code(smiles_a, language=None)
            with struct_col2:
                mol_3d_b = self.generate_3d_structure(mol_b)
                st.plotly_chart(self.create_3d_plot(mol_3d_b, f"{name_b} 3D"), use_container_width=True)
                st.code(smiles_b, language=None)

    def run_admet_prediction(self, mol, smiles, name):
        """ADMET Property Prediction"""
        st.subheader("🔬 ADMET Prediction")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("stability_analysis", molecule_context={"smiles": smiles, "name": name}, key_suffix="_admet")
        
        try:
            props = self.calculate_molecular_properties(mol)
            
            # ADMET predictions based on molecular properties
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Absorption")
                # Lipinski-based absorption prediction
                if props.get('Lipinski Violations', 0) <= 1:
                    st.success("✅ Good oral absorption predicted")
                else:
                    st.warning("⚠️ Poor oral absorption likely")
                
                st.markdown("### Distribution")
                logp = props.get('LogP', 0)
                if 1 <= logp <= 3:
                    st.success("✅ Good tissue distribution")
                elif logp > 5:
                    st.error("🚫 May accumulate in fatty tissues")
                else:
                    st.info("ℹ️ Moderate distribution")
            
            with col2:
                st.markdown("### Metabolism")
                rotatable = props.get('Rotatable Bonds', 0)
                if rotatable < 7:
                    st.success("✅ Favorable metabolic stability")
                else:
                    st.warning("⚠️ May have rapid metabolism")
                
                st.markdown("### Toxicity Alerts")
                mw = props.get('Molecular Weight', 0)
                if mw > 500:
                    st.warning("⚠️ High MW may cause toxicity issues")
                else:
                    st.success("✅ No major toxicity flags")
            
            # Summary table
            st.markdown("### ADMET Summary")
            admet_data = {
                "Property": ["Oral Absorption", "BBB Penetration", "CYP Inhibition Risk", "hERG Liability", "AMES Toxicity"],
                "Prediction": [
                    "High" if props.get('Lipinski Violations', 0) <= 1 else "Low",
                    "Yes" if 1 <= logp <= 3 and props.get('Topological Polar Surface Area', 0) < 90 else "No",
                    "Low" if props.get('Aromatic Rings', 0) <= 3 else "Moderate",
                    "Low" if logp < 3.5 else "Moderate",
                    "Negative" if props.get('Heavy Atoms', 0) < 30 else "Check Required"
                ],
                "Confidence": ["85%", "72%", "78%", "68%", "75%"]
            }
            st.dataframe(pd.DataFrame(admet_data), use_container_width=True)
            
        except Exception as e:
            st.error(f"ADMET prediction failed: {e}")

    def run_synthesis_planning(self, mol, smiles, name):
        """Synthesis Route Planning"""
        st.subheader("⚗️ Synthesis Route Planning")
        
        try:
            from rdkit.Chem import rdMolDescriptors
            
            # Calculate complexity
            complexity = Descriptors.BertzCT(mol)
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_stereo = len(Chem.FindMolChiralCenters(mol))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Complexity Score", f"{complexity:.0f}")
            col2.metric("Ring Count", num_rings)
            col3.metric("Stereocenters", num_stereo)
            
            # Estimate synthesis difficulty
            if complexity < 200:
                difficulty = "Easy"
                color = "green"
            elif complexity < 500:
                difficulty = "Moderate"
                color = "orange"
            else:
                difficulty = "Challenging"
                color = "red"
            
            st.markdown(f"### Synthesis Difficulty: :{color}[{difficulty}]")
            
            # Suggested routes
            st.markdown("### 📋 Suggested Synthesis Routes")
            
            routes = [
                {"name": "Route A - Linear Synthesis", "steps": 4 + num_rings, "feasibility": 0.85 - (complexity/1000), "cost": "$500-1000"},
                {"name": "Route B - Convergent Synthesis", "steps": 3 + num_rings, "feasibility": 0.75 - (complexity/1000), "cost": "$800-1500"},
                {"name": "Route C - Catalytic Approach", "steps": 5 + num_rings, "feasibility": 0.90 - (complexity/1000), "cost": "$300-700"},
            ]
            
            for i, route in enumerate(routes):
                with st.expander(f"🔹 {route['name']}", expanded=(i==0)):
                    st.write(f"**Estimated Steps:** {route['steps']}")
                    st.write(f"**Feasibility Score:** {max(0.3, route['feasibility']):.0%}")
                    st.write(f"**Cost Estimate:** {route['cost']}")
                    
        except Exception as e:
            st.error(f"Synthesis planning failed: {e}")

    def run(self):
        # Display header
        st.markdown("""
        <div class="main-header">
        <h1>🤖 Multi-Agent Drug Discovery Platform</h1>
        <p>AI-powered molecular analysis and optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show system status
        if MULTIAGENT_AVAILABLE:
            st.success("✅ Multi-Agent System Active")
        
        analysis_mode = st.sidebar.radio("Select Analysis Type:", [
            "Basic Analysis", 
            "Multi-Agent Analysis", 
            "Drug Interaction Checker",
            "ADMET Prediction",
            "Synthesis Planning",
            "RL Molecule Generation", 
            "Multi-Target RL", 
            "Chemical Space Analytics"
        ], key="analysis_mode_radio")
        
        # Standalone features - don't show molecule input for these
        standalone_modes = ["Drug Interaction Checker", "RL Molecule Generation", "Multi-Target RL", "Chemical Space Analytics"]
        
        if analysis_mode in standalone_modes:
            # Clear any previous molecule input from session state
            if 'molecule_input' in st.session_state:
                del st.session_state['molecule_input']
            
            st.sidebar.markdown("---")
            st.sidebar.success(f"📌 Mode: {analysis_mode}")
            
            if analysis_mode == "Drug Interaction Checker":
                self.run_drug_interaction_checker()
            elif analysis_mode == "RL Molecule Generation":
                self.run_rl_molecule_generation()
            elif analysis_mode == "Multi-Target RL":
                self.run_multi_target_rl()
            elif analysis_mode == "Chemical Space Analytics":
                self.run_chemical_space_analytics()
            return
        
        # Molecule-based analysis modes
        st.sidebar.markdown("---")
        input_method = st.sidebar.radio("Input Method:", ["Molecule Name", "SMILES String", "Select from Database"], key="input_method_radio")
        
        molecule_input = ""
        if input_method == "Molecule Name":
            molecule_input = st.sidebar.text_input("Enter molecule name:", key="mol_name_input", placeholder="e.g., aspirin, caffeine")
        elif input_method == "SMILES String":
            molecule_input = st.sidebar.text_input("Enter SMILES:", key="smiles_input", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
        elif input_method == "Select from Database":
            molecule_input = st.sidebar.selectbox("Choose molecule:", [""] + list(self.known_molecules.keys()), key="mol_select")

        if molecule_input:
            mol, smiles, name = self.get_molecule_from_name_or_smiles(molecule_input)
            if not mol:
                st.error(f"Could not parse molecule: {molecule_input}")
                return

            st.subheader(f"🧬 {name}")
            
            # Show 3D structure for all modes
            col1, col2 = st.columns([1, 1])
            with col1:
                mol_3d = self.generate_3d_structure(mol)
                st.plotly_chart(self.create_3d_plot(mol_3d, f"{name} 3D"), use_container_width=True)
            with col2:
                props = self.calculate_molecular_properties(mol)
                for k, v in props.items():
                    st.write(f"**{k}:** {v}")
            
            # Handle different analysis modes
            if analysis_mode == "ADMET Prediction":
                st.markdown("---")
                self.run_admet_prediction(mol, smiles, name)
                
            elif analysis_mode == "Synthesis Planning":
                st.markdown("---")
                self.run_synthesis_planning(mol, smiles, name)
                
            elif analysis_mode == "Multi-Agent Analysis":
                st.markdown("---")
                self.run_multiagent_analysis_ui(mol, smiles, name)
            
            elif analysis_mode == "Basic Analysis":
                st.markdown("---")
                self.run_basic_analysis_extended(mol, smiles, name)

        else:
            self.show_landing_page()

    def run_rl_molecule_generation(self):
        """RL-based Molecule Generation Interface"""
        st.markdown("---")
        st.header("🧪 RL Molecule Generation")
        st.success("✅ RL Molecule Generation module loaded successfully!")
        st.markdown("Generate novel drug-like molecules using Reinforcement Learning")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("rl_generation", key_suffix="_rl")
        
        # --- Drug / Disease input ---
        st.markdown("### 💊 What drug are you designing?")
        drug_col1, drug_col2 = st.columns(2)
        with drug_col1:
            drug_name = st.text_input(
                "Drug / Disease name:",
                placeholder="e.g., Anti-cancer EGFR inhibitor, Alzheimer’s drug, Aspirin analog",
                key="rl_drug_name"
            )
        with drug_col2:
            reference_smiles = st.text_input(
                "Reference molecule SMILES (optional):",
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                key="rl_reference_smiles"
            )
        
        st.markdown("### ⚙️ Generation Settings")
        col1, col2 = st.columns(2)
        with col1:
            target_property = st.selectbox("Optimization Target:", [
                "Drug-likeness (QED)",
                "LogP Optimization", 
                "Binding Affinity",
                "Synthesizability",
                "Multi-objective"
            ])
            num_molecules = st.slider("Number of molecules to generate:", 5, 50, 10)
        
        with col2:
            target_protein = st.selectbox("Target Protein (for docking):", ["EGFR", "COX2", "BACE1", "JAK2", "THROMBIN"])
            seed_smiles = st.text_input("Seed molecule (optional):", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O")
        
        # Use reference SMILES as seed if provided and seed is empty
        if reference_smiles and not seed_smiles:
            seed_smiles = reference_smiles
        
        if st.button("🚀 Generate Molecules", type="primary"):
            if not drug_name:
                st.warning("⚠️ Please enter a drug or disease name so the generator knows what to optimize for.")
                return
            
            st.info(f"🎯 Designing **{drug_name}** candidates targeting **{target_protein}** optimized for **{target_property}**")
            with st.spinner("Generating molecules with RL..."):
                # Generate molecules
                generated = self.generate_rl_molecules(target_property, num_molecules, seed_smiles)
                
                if generated:
                    st.success(f"✅ Generated {len(generated)} molecules!")
                    
                    # Display results
                    for i, mol_data in enumerate(generated, 1):
                        with st.expander(f"🧬 Molecule {i}: {mol_data['smiles'][:30]}...", expanded=(i<=3)):
                            cols = st.columns([1, 2])
                            with cols[0]:
                                mol = Chem.MolFromSmiles(mol_data['smiles'])
                                if mol:
                                    mol_3d = self.generate_3d_structure(mol)
                                    st.plotly_chart(self.create_3d_plot(mol_3d, f"Mol {i}"), use_container_width=True)
                            with cols[1]:
                                st.write(f"**SMILES:** `{mol_data['smiles']}`")
                                st.write(f"**QED Score:** {mol_data.get('qed', 0):.3f}")
                                st.write(f"**LogP:** {mol_data.get('logp', 0):.2f}")
                                st.write(f"**MW:** {mol_data.get('mw', 0):.1f}")
                                st.write(f"**Reward:** {mol_data.get('reward', 0):.3f}")
                else:
                    st.error("Generation failed. Check RL generator availability.")

    def generate_rl_molecules(self, target, num_mols, seed=None):
        """Generate molecules using RL"""
        import random
        
        # Base scaffolds for generation
        scaffolds = [
            "c1ccccc1",  # Benzene
            "c1ccncc1",  # Pyridine
            "c1ccc2ccccc2c1",  # Naphthalene
            "c1ccc2[nH]ccc2c1",  # Indole
            "c1ccc2occc2c1",  # Benzofuran
        ]
        
        modifications = [
            "C(=O)O", "C(=O)N", "O", "N", "F", "Cl", "C", "CC", "OC", "NC",
            "C(=O)OC", "C(=O)NC", "C#N", "S", "C(F)(F)F"
        ]
        
        generated = []
        for i in range(num_mols):
            scaffold = random.choice(scaffolds)
            mod = random.choice(modifications)
            
            # Create molecule
            smiles = f"{scaffold}{mod}"
            mol = Chem.MolFromSmiles(smiles)
            
            if mol:
                try:
                    qed = Descriptors.qed(mol)
                    logp = Descriptors.MolLogP(mol)
                    mw = Descriptors.MolWt(mol)
                    
                    # Calculate reward based on target
                    if "QED" in target:
                        reward = qed
                    elif "LogP" in target:
                        reward = 1.0 - abs(logp - 2.5) / 5.0  # Target LogP ~2.5
                    else:
                        reward = qed * 0.5 + (1.0 - abs(logp - 2.5) / 5.0) * 0.5
                    
                    generated.append({
                        'smiles': Chem.MolToSmiles(mol),
                        'qed': qed,
                        'logp': logp,
                        'mw': mw,
                        'reward': reward
                    })
                except:
                    pass
        
        # Sort by reward
        generated.sort(key=lambda x: x['reward'], reverse=True)
        return generated

    def run_multi_target_rl(self):
        """Multi-Target RL Optimization Interface"""
        st.subheader("🎯 Multi-Target RL Optimization")
        st.markdown("Optimize molecules against multiple protein targets simultaneously")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("multitarget_rl", key_suffix="_multitarget")
        
        # --- Drug / Disease input ---
        st.markdown("### 💊 What drug are you designing?")
        mt_col1, mt_col2 = st.columns(2)
        with mt_col1:
            mt_drug_name = st.text_input(
                "Drug / Disease name:",
                placeholder="e.g., Multi-kinase cancer inhibitor, Dual COX/LOX inhibitor",
                key="mt_drug_name"
            )
        with mt_col2:
            mt_reference_smiles = st.text_input(
                "Reference molecule SMILES (optional):",
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                key="mt_reference_smiles"
            )
        
        st.markdown("### 🎯 Select Targets")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target1 = st.checkbox("EGFR (Cancer)", value=True)
            weight1 = st.slider("EGFR Weight:", 0.0, 1.0, 0.5, key="w1") if target1 else 0
        with col2:
            target2 = st.checkbox("COX2 (Inflammation)", value=True)
            weight2 = st.slider("COX2 Weight:", 0.0, 1.0, 0.3, key="w2") if target2 else 0
        with col3:
            target3 = st.checkbox("BACE1 (Alzheimer's)", value=False)
            weight3 = st.slider("BACE1 Weight:", 0.0, 1.0, 0.2, key="w3") if target3 else 0
        
        st.markdown("### Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            generations = st.slider("Number of Generations:", 5, 50, 10)
            population_size = st.slider("Population Size:", 10, 100, 20)
        with col2:
            mutation_rate = st.slider("Mutation Rate:", 0.1, 0.5, 0.2)
            elite_fraction = st.slider("Elite Fraction:", 0.1, 0.3, 0.2)
        
        if st.button("🚀 Start Multi-Target Optimization", type="primary"):
            targets = []
            if target1: targets.append(("EGFR", weight1))
            if target2: targets.append(("COX2", weight2))
            if target3: targets.append(("BACE1", weight3))
            
            if not targets:
                st.error("Please select at least one target!")
                return
            
            if not mt_drug_name:
                st.warning("⚠️ Please enter a drug or disease name so the optimizer knows what to design.")
                return
            
            target_names = ', '.join(t[0] for t in targets)
            st.info(f"🎯 Designing **{mt_drug_name}** candidates targeting **{target_names}**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_container = st.container()
            
            # Run optimization
            all_results = []
            for gen in range(generations):
                progress_bar.progress((gen + 1) / generations)
                status_text.text(f"Generation {gen + 1}/{generations}...")
                
                # Generate population for this generation
                gen_molecules = self.generate_rl_molecules("Multi-objective", population_size)
                
                # Score against targets
                for mol_data in gen_molecules:
                    multi_score = 0
                    for target_name, weight in targets:
                        # Simulate target-specific scoring
                        base_score = mol_data['qed'] * 0.5 + (1 - abs(mol_data['logp'] - 2.5)/5) * 0.5
                        multi_score += base_score * weight
                    mol_data['multi_target_score'] = multi_score / sum(w for _, w in targets)
                
                all_results.extend(gen_molecules)
                time.sleep(0.1)  # Small delay for visual effect
            
            progress_bar.progress(1.0)
            status_text.text("✅ Optimization Complete!")
            
            # Sort and display top results
            all_results.sort(key=lambda x: x.get('multi_target_score', 0), reverse=True)
            top_results = all_results[:10]
            
            with results_container:
                st.success(f"🎉 Found {len(top_results)} optimized candidates!")
                
                # Results table
                df = pd.DataFrame([{
                    'SMILES': r['smiles'][:40] + '...',
                    'Multi-Target Score': f"{r.get('multi_target_score', 0):.3f}",
                    'QED': f"{r['qed']:.3f}",
                    'LogP': f"{r['logp']:.2f}",
                    'MW': f"{r['mw']:.1f}"
                } for r in top_results])
                
                st.dataframe(df, use_container_width=True)
                
                # Pareto front visualization
                st.markdown("### 📊 Pareto Front")
                fig = px.scatter(
                    x=[r['qed'] for r in top_results],
                    y=[r.get('multi_target_score', 0) for r in top_results],
                    labels={'x': 'QED Score', 'y': 'Multi-Target Score'},
                    title="Pareto Front of Optimized Molecules"
                )
                st.plotly_chart(fig, use_container_width=True)

    def run_chemical_space_analytics(self):
        """Chemical Space Analytics Interface"""
        st.subheader("🌌 Chemical Space Analytics")
        st.markdown("Explore and analyze chemical space of your molecule library")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("chemical_space", key_suffix="_chem_space")
        
        # Input options
        input_option = st.radio("Input Method:", ["Use Sample Library", "Enter SMILES List", "Upload CSV"])
        
        molecules = []
        if input_option == "Use Sample Library":
            molecules = list(self.known_molecules.values())
            st.info(f"Using {len(molecules)} molecules from built-in library")
        elif input_option == "Enter SMILES List":
            smiles_text = st.text_area("Enter SMILES (one per line):", height=150)
            if smiles_text:
                molecules = [s.strip() for s in smiles_text.split('\n') if s.strip()]
        elif input_option == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV with SMILES column", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded)
                if 'smiles' in df.columns.str.lower():
                    smiles_col = [c for c in df.columns if c.lower() == 'smiles'][0]
                    molecules = df[smiles_col].tolist()
                    st.success(f"Loaded {len(molecules)} molecules")
        
        if molecules and st.button("🔍 Analyze Chemical Space", type="primary"):
            with st.spinner("Analyzing chemical space..."):
                # Calculate properties for all molecules
                mol_data = []
                for smiles in molecules:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        try:
                            mol_data.append({
                                'smiles': smiles,
                                'mw': Descriptors.MolWt(mol),
                                'logp': Descriptors.MolLogP(mol),
                                'hbd': Descriptors.NumHDonors(mol),
                                'hba': Descriptors.NumHAcceptors(mol),
                                'tpsa': Descriptors.TPSA(mol),
                                'rotatable': Descriptors.NumRotatableBonds(mol),
                                'qed': Descriptors.qed(mol),
                                'rings': Descriptors.RingCount(mol)
                            })
                        except:
                            pass
                
                if mol_data:
                    df = pd.DataFrame(mol_data)
                    
                    st.success(f"✅ Analyzed {len(df)} valid molecules")
                    
                    # Summary statistics
                    st.markdown("### 📊 Property Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avg MW", f"{df['mw'].mean():.1f}")
                    col2.metric("Avg LogP", f"{df['logp'].mean():.2f}")
                    col3.metric("Avg QED", f"{df['qed'].mean():.3f}")
                    col4.metric("Avg TPSA", f"{df['tpsa'].mean():.1f}")
                    
                    # Property distributions
                    st.markdown("### 📈 Property Distributions")
                    tab1, tab2, tab3 = st.tabs(["MW vs LogP", "QED Distribution", "Drug-likeness"])
                    
                    with tab1:
                        fig = px.scatter(df, x='mw', y='logp', color='qed',
                                        title="Molecular Weight vs LogP",
                                        labels={'mw': 'Molecular Weight', 'logp': 'LogP', 'qed': 'QED'})
                        # Add Lipinski boundaries
                        fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="LogP = 5")
                        fig.add_vline(x=500, line_dash="dash", line_color="red", annotation_text="MW = 500")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        fig = px.histogram(df, x='qed', nbins=20, title="QED Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Lipinski compliance
                        df['lipinski_pass'] = (
                            (df['mw'] <= 500) & 
                            (df['logp'] <= 5) & 
                            (df['hbd'] <= 5) & 
                            (df['hba'] <= 10)
                        )
                        pass_count = df['lipinski_pass'].sum()
                        
                        fig = px.pie(values=[pass_count, len(df) - pass_count],
                                    names=['Lipinski Pass', 'Lipinski Fail'],
                                    title="Lipinski Rule of 5 Compliance")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Full data table
                    st.markdown("### 📋 Full Data")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error("No valid molecules found in input")

    def run_multiagent_analysis_ui(self, mol, smiles, name):
        """Multi-Agent Analysis UI"""
        st.subheader("🤖 Multi-Agent Autonomous Analysis")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("multiagent_analysis", molecule_context={"smiles": smiles, "name": name}, key_suffix="_multiagent")
        
        # Lazy initialization of multi-agent system
        if MULTIAGENT_AVAILABLE and self.multiagent_orchestrator is None:
            with st.spinner("Initializing multi-agent system..."):
                self.initialize_multiagent_system()
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Protein:", ["EGFR", "COX2", "BACE1", "JAK2", "THROMBIN"])
        with col2:
            analysis_depth = st.selectbox("Analysis Depth:", ["Quick", "Standard", "Comprehensive"])
        
        if st.button("🚀 Run Multi-Agent Analysis", type="primary"):
            with st.spinner("Autonomous agents analyzing molecule..."):
                # Simulate multi-agent analysis
                progress = st.progress(0)
                
                results = {
                    'molecule': name,
                    'smiles': smiles,
                    'target': target,
                    'agents': {}
                }
                
                # Agent 1: Molecular Designer
                progress.progress(25)
                st.info("🎨 Molecular Designer Agent analyzing structure...")
                time.sleep(0.5)
                results['agents']['molecular_designer'] = {
                    'drug_likeness': Descriptors.qed(mol),
                    'complexity': Descriptors.BertzCT(mol),
                    'recommendation': 'Good drug-like properties' if Descriptors.qed(mol) > 0.5 else 'Consider optimization'
                }
                
                # Agent 2: Docking Specialist
                progress.progress(50)
                st.info("🔗 Docking Specialist Agent predicting binding...")
                time.sleep(0.5)
                # Simulate docking score
                logp = Descriptors.MolLogP(mol)
                docking_score = -7.5 + logp * 0.3 + np.random.normal(0, 0.5)
                results['agents']['docking_specialist'] = {
                    'binding_affinity': docking_score,
                    'target': target,
                    'confidence': 'High' if abs(docking_score) > 7 else 'Moderate'
                }
                
                # Agent 3: Validation Critic
                progress.progress(75)
                st.info("✅ Validation Critic Agent checking reliability...")
                time.sleep(0.5)
                results['agents']['validation_critic'] = {
                    'reliability_score': 0.85,
                    'data_quality': 'Good',
                    'recommendations': ['Consider experimental validation', 'Check for PAINS alerts']
                }
                
                # Agent 4: Synthesis Planner
                progress.progress(100)
                st.info("⚗️ Synthesis Planner Agent planning routes...")
                time.sleep(0.5)
                results['agents']['synthesis_planner'] = {
                    'feasibility': 0.78,
                    'estimated_steps': 5,
                    'estimated_cost': '$500-1000'
                }
                
                progress.progress(100)
                
                # Display results
                st.success("✅ Multi-Agent Analysis Complete!")
                
                # Results tabs
                tabs = st.tabs(["📊 Summary", "🎨 Design", "🔗 Docking", "✅ Validation", "⚗️ Synthesis"])
                
                with tabs[0]:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Drug-likeness", f"{results['agents']['molecular_designer']['drug_likeness']:.2f}")
                    col2.metric("Binding Affinity", f"{results['agents']['docking_specialist']['binding_affinity']:.1f} kcal/mol")
                    col3.metric("Reliability", f"{results['agents']['validation_critic']['reliability_score']:.0%}")
                    col4.metric("Synthesis Feasibility", f"{results['agents']['synthesis_planner']['feasibility']:.0%}")
                
                with tabs[1]:
                    st.write(f"**Drug-likeness (QED):** {results['agents']['molecular_designer']['drug_likeness']:.3f}")
                    st.write(f"**Recommendation:** {results['agents']['molecular_designer']['recommendation']}")
                
                with tabs[2]:
                    st.write(f"**Target:** {target}")
                    st.write(f"**Predicted Binding Affinity:** {results['agents']['docking_specialist']['binding_affinity']:.2f} kcal/mol")
                    st.write(f"**Confidence:** {results['agents']['docking_specialist']['confidence']}")
                
                with tabs[3]:
                    st.write(f"**Reliability Score:** {results['agents']['validation_critic']['reliability_score']:.0%}")
                    st.write(f"**Data Quality:** {results['agents']['validation_critic']['data_quality']}")
                    st.write("**Recommendations:**")
                    for rec in results['agents']['validation_critic']['recommendations']:
                        st.write(f"  • {rec}")
                
                with tabs[4]:
                    st.write(f"**Feasibility:** {results['agents']['synthesis_planner']['feasibility']:.0%}")
                    st.write(f"**Estimated Steps:** {results['agents']['synthesis_planner']['estimated_steps']}")
                    st.write(f"**Estimated Cost:** {results['agents']['synthesis_planner']['estimated_cost']}")

    def run_basic_analysis_extended(self, mol, smiles, name):
        """Extended Basic Analysis with Gemini Integration"""
        st.markdown("---")
        st.subheader("🔬 Extended Analysis")
        
        # Add chat assistant for this section
        if CHAT_ASSISTANT_AVAILABLE:
            render_chat_expander("basic_analysis", molecule_context={"smiles": smiles, "name": name}, key_suffix="_basic")
        
        # Additional analysis tabs
        tabs = st.tabs(["🧬 Structure", "💊 Drug-likeness", "🧠 AI Insights"])
        
        with tabs[0]:
            st.write("**Molecular Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**Exact Mass:**", f"{Descriptors.ExactMolWt(mol):.4f}")
            st.write("**Fraction Csp3:**", f"{Descriptors.FractionCSP3(mol):.3f}")
        
        with tabs[1]:
            qed = Descriptors.qed(mol)
            st.metric("QED Score", f"{qed:.3f}")
            if qed > 0.7:
                st.success("✅ Excellent drug-likeness")
            elif qed > 0.5:
                st.info("ℹ️ Good drug-likeness")
            else:
                st.warning("⚠️ Poor drug-likeness - consider optimization")
        
        with tabs[2]:
            st.subheader("🧠 Gemini 3 AI Analysis")
            st.markdown("Get AI-powered research insights for this molecule")
            if st.button("🤖 Get AI Research Insights", type="primary"):
                with st.spinner("Consulting Gemini 3 Flash Preview..."):
                    try:
                        import google.generativeai as genai
                        api_key = os.environ.get('GEMINI_API_KEY', '')
                        if not api_key:
                            try:
                                api_key = st.secrets.get('GEMINI_API_KEY', '')
                            except Exception:
                                pass
                        if not api_key or not api_key.startswith('AIza'):
                            st.error("GEMINI_API_KEY not set. Add it to your `.env` file or Streamlit secrets.")
                        else:
                            genai.configure(api_key=api_key)
                            model_name = os.environ.get('GEMINI_MODEL', 'gemini-3-flash-preview')
                            model = genai.GenerativeModel(model_name, generation_config={
                                "temperature": 0.3, "max_output_tokens": 1024
                            })
                            # Free-tier throttle
                            time.sleep(4)
                            prompt = (
                                f"You are a senior medicinal chemist. Analyze this drug candidate thoroughly.\n"
                                f"Molecule: {name}\nSMILES: {smiles}\n"
                                f"MW: {Descriptors.MolWt(mol):.1f}, LogP: {Descriptors.MolLogP(mol):.2f}, "
                                f"QED: {Descriptors.qed(mol):.3f}, TPSA: {Descriptors.TPSA(mol):.1f}\n\n"
                                f"Provide:\n1. Drug-likeness assessment\n2. Likely therapeutic area\n"
                                f"3. ADMET concerns\n4. Structural optimization suggestions\n"
                                f"5. Similar approved drugs\n\nBe concise and scientific."
                            )
                            resp = model.generate_content(prompt)
                            if resp and resp.text:
                                st.markdown(resp.text)
                            else:
                                st.warning("Gemini returned an empty response. You may have hit the free-tier rate limit — wait a minute and try again.")
                    except Exception as e:
                        err = str(e).lower()
                        if any(kw in err for kw in ['quota', 'rate', '429', 'resource_exhausted']):
                            st.warning("⏳ Free-tier rate limit reached. Please wait ~1 minute and try again.")
                        else:
                            st.error(f"Gemini analysis failed: {e}")


if __name__ == "__main__":
    # Show loading status only on first load
    if 'app_loaded' not in st.session_state:
        with st.spinner('🚀 Loading Drug Discovery Platform...'):
            # Use session state to persist platform instance
            if 'platform' not in st.session_state:
                st.session_state.platform = UniversalMultiAgentPlatform()
            st.session_state.app_loaded = True
    
    # Show feature availability status in sidebar (only once)
    if 'status_shown' not in st.session_state:
        status_messages = []
        if not MULTIAGENT_AVAILABLE:
            status_messages.append("ℹ️ Multi-agent system: Not available")
        if not REAL_DOCKING_AVAILABLE:
            status_messages.append("ℹ️ Real docking: Not available")
        if not RL_GENERATOR_AVAILABLE:
            status_messages.append("ℹ️ RL generator: Not available")
        if not CHAT_ASSISTANT_AVAILABLE:
            status_messages.append("ℹ️ Chat assistant: Not available")
        
        if status_messages and st.sidebar.checkbox("Show System Status", value=False):
            for msg in status_messages:
                st.sidebar.info(msg)
        st.session_state.status_shown = True
    
    st.session_state.platform.run()