#!/usr/bin/env python3
"""
Advanced Molecular Stability & Safety Analysis System

Comprehensive analysis of molecular stability including:
    - Thermodynamic stability prediction
    - Kinetic stability assessment
    - Reactivity analysis
    - Drug-like property evaluation
    - Toxicity prediction
    - ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) analysis
    """

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
    warnings.filterwarnings('ignore')

    print(" Initializing Advanced Stability Analysis...")

# Auto-install required packages
    def install_if_missing(package):
        try:
            __import__(package.split('[')[0])
            print(f" {package}")
        except ImportError:
            print(f"📥 Installing {package}...")
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package])

            required_packages = [
                "rdkit",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "plotly",
                "mordred",
                "chembl-webresource-client"
            ]

            for pkg in required_packages:
                install_if_missing(pkg)

# Import dependencies
                from rdkit import Chem
                from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Crippen
                from rdkit.Chem.Scaffolds import MurckoScaffold
                from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
                from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAromaticRings
                import scipy.constants
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                import matplotlib.pyplot as plt
                import seaborn as sns
                import plotly.graph_objects as go
                import plotly.express as px

                class StabilityRisk(Enum):
                    """Molecular stability risk levels"""
                    VERY_LOW = ("Very Low Risk", "#28a745")  # Green
                    LOW = ("Low Risk", "#6c757d")  # Blue
                    MODERATE = ("Moderate Risk", "#ffc107")  # Yellow
                    HIGH = ("High Risk", "#fd7e14")  # Orange
                    VERY_HIGH = ("Very High Risk", "#dc3545")  # Red
                    CRITICAL = ("Critical Risk", "#6f42c1")  # Purple

                    class ToxicityLevel(Enum):
                        """Toxicity classification levels"""
                        NON_TOXIC = ("Non-Toxic", "#28a745")
                        MILD = ("Mild Toxicity", "#17a2b8")
                        MODERATE = ("Moderate Toxicity", "#ffc107")
                        HIGH = ("High Toxicity", "#fd7e14")
                        SEVERE = ("Severe Toxicity", "#dc3545")

                        @dataclass
                        class StabilityAnalysis:
                            """Complete stability analysis results"""
    # Thermodynamic properties
                            molecular_weight: float
                            logp: float
                            tpsa: float
                            formal_charge: int

    # Structural features
                            num_rings: int
                            aromatic_rings: int
                            rotatable_bonds: int
                            hb_donors: int
                            hb_acceptors: int

    # Stability metrics
                            lipinski_violations: int
                            veber_violations: int
                            egan_violations: int
                            reactive_groups: List[str]

    # Predictive scores (0-100)
                            thermodynamic_stability: float
                            kinetic_stability: float
                            metabolic_stability: float
                            photostability: float
                            overall_stability: float

    # Risk assessment
                            stability_risk: StabilityRisk
                            toxicity_risk: ToxicityLevel

    # ADMET predictions
                            absorption: float  # 0-100
                            distribution: float
                            metabolism: float
                            excretion: float
                            toxicity: float

    # Recommendations
                            recommendations: List[str]
                            warnings: List[str]

                            class AdvancedStabilityAnalyzer:
                                """Advanced molecular stability and safety analysis system"""

                                def __init__(self):
                                    self.reactive_patterns = self._initialize_reactive_patterns()
                                    self.filter_catalog = self._setup_filter_catalog()
                                    self.models = {}
                                    print(
                                        " Advanced Stability Analyzer initialized")

                                    def _initialize_reactive_patterns(
                                            self) -> Dict[str, str]:
                                        """Initialize reactive group patterns"""
                                        return {
                                            "aldehyde": "[CX3H1](=O)",
                                            "ketone": "[#6][CX3](=O)[#6]",
                                            "ester": "[CX3](=O)[OX2H0][#6]",
                                            "amide": "[CX3](=O)[NX3H2,NX3H1,NX3H0]",
                                            "nitrile": "[CX2]#N",
                                            "nitro": "[NX3](=O)=O",
                                            "epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
                                            "Michael_acceptor": "[CX3]=[CX3]-[CX3](=O)",
                                            "alkyl_halide": "[CX4][F,Cl,Br,I]",
                                            "phenol": "[OX2H][cX3]:[c]",
                                            "aniline": "[NX3H2,NX3H1][cX3]:[c]",
                                            "thiol": "[SX2H]",
                                            "disulfide": "[SX2][SX2]",
                                            "peroxide": "[OX2][OX2]",
                                            "azide": "[NX2]=[NX2]=[NX1]",
                                            "diazonium": "[NX2+]#[NX1]",
                                            "acid_chloride": "[CX3](=O)[ClX1]",
                                            "isocyanate": "[NX2]=[CX2]=[OX1]",
                                            "hydrazine": "[NX3H2,NX3H1][NX3H2,NX3H1]",
                                            "hydroxylamine": "[NX3H2,NX3H1][OX2H,OX2]"}

                                    def _setup_filter_catalog(
                                            self) -> FilterCatalog:
                                        """Setup RDKit filter catalog for reactive groups"""
                                        params = FilterCatalogParams()
                                        params.AddCatalog(
                                            FilterCatalogParams.FilterCatalogs.PAINS)
                                        params.AddCatalog(
                                            FilterCatalogParams.FilterCatalogs.BRENK)
                                        return FilterCatalog(params)

                                    def analyze_molecule(
                                            self, molecule_input: Union[str, Chem.Mol]) -> StabilityAnalysis:
                                        """Perform comprehensive stability analysis"""

    # Handle input
                                        if isinstance(molecule_input, str):
                                            mol = Chem.MolFromSmiles(
                                                molecule_input)
                                            smiles = molecule_input
                                        else:
                                            mol = molecule_input
                                            smiles = Chem.MolToSmiles(mol)

                                            if mol is None:
                                                raise ValueError(
             "Invalid molecule input")

    # Calculate basic properties
                                            mw = Descriptors.MolWt(mol)
                                            logp = Crippen.MolLogP(mol)
                                            tpsa = Descriptors.TPSA(mol)
                                            formal_charge = Chem.rdmolops.GetFormalCharge(
                                                mol)

    # Structural features
                                            num_rings = CalcNumRings(mol)
                                            aromatic_rings = CalcNumAromaticRings(
                                                mol)
                                            rotatable_bonds = Descriptors.NumRotatableBonds(
                                                mol)
                                            hb_donors = Descriptors.NumHDonors(
                                                mol)
                                            hb_acceptors = Descriptors.NumHAcceptors(
                                                mol)

    # Drug-like property violations
                                            lipinski_violations = self._calculate_lipinski_violations(
                                                mw, logp, hb_donors, hb_acceptors)
                                            veber_violations = self._calculate_veber_violations(
                                                rotatable_bonds, tpsa)
                                            egan_violations = self._calculate_egan_violations(
                                                logp, tpsa)

    # Reactive groups analysis
                                            reactive_groups = self._detect_reactive_groups(
                                                mol)

    # Stability predictions
                                            thermo_stability = self._predict_thermodynamic_stability(
                                                mol, mw, logp, formal_charge)
                                            kinetic_stability = self._predict_kinetic_stability(
                                                mol, reactive_groups, aromatic_rings)
                                            metabolic_stability = self._predict_metabolic_stability(
                                                mol, logp, tpsa)
                                            photostability = self._predict_photostability(
                                                mol, aromatic_rings)

    # Overall stability score
                                            overall_stability = (
                                                thermo_stability + kinetic_stability + metabolic_stability + photostability) / 4

    # Risk assessment
                                            stability_risk = self._assess_stability_risk(
                                                overall_stability, reactive_groups)
                                            toxicity_risk = self._assess_toxicity_risk(
                                                mol, reactive_groups)

    # ADMET predictions
                                            admet = self._predict_admet(
                                                mol, mw, logp, tpsa, hb_donors, hb_acceptors)

    # Generate recommendations and warnings
                                            recommendations, warnings = self._generate_recommendations(
                                                mol, overall_stability, reactive_groups, lipinski_violations
                                            )

                                            return StabilityAnalysis(
                                                molecular_weight=mw,
                                                logp=logp,
                                                tpsa=tpsa,
                                                formal_charge=formal_charge,
                                                num_rings=num_rings,
                                                aromatic_rings=aromatic_rings,
                                                rotatable_bonds=rotatable_bonds,
                                                hb_donors=hb_donors,
                                                hb_acceptors=hb_acceptors,
                                                lipinski_violations=lipinski_violations,
                                                veber_violations=veber_violations,
                                                egan_violations=egan_violations,
                                                reactive_groups=reactive_groups,
                                                thermodynamic_stability=thermo_stability,
                                                kinetic_stability=kinetic_stability,
                                                metabolic_stability=metabolic_stability,
                                                photostability=photostability,
                                                overall_stability=overall_stability,
                                                stability_risk=stability_risk,
                                                toxicity_risk=toxicity_risk,
                                                absorption=admet['absorption'],
                                                distribution=admet['distribution'],
                                                metabolism=admet['metabolism'],
                                                excretion=admet['excretion'],
                                                toxicity=admet['toxicity'],
                                                recommendations=recommendations,
                                                warnings=warnings
                                            )

                                        def _calculate_lipinski_violations(
                                                self, mw: float, logp: float, hb_donors: int, hb_acceptors: int) -> int:
                                            """Calculate Lipinski Rule of Five violations"""
                                            violations = 0
                                            if mw > 500:
                                                violations += 1
                                                if logp > 5:
             violations += 1
             if hb_donors > 5:
              violations += 1
              if hb_acceptors > 10:
               violations += 1
               return violations

              def _calculate_veber_violations(
                self, rotatable_bonds: int, tpsa: float) -> int:
               """Calculate Veber rule violations"""
               violations = 0
               if rotatable_bonds > 10:
                violations += 1
                if tpsa > 140:
                 violations += 1
                 return violations

                def _calculate_egan_violations(
                  self, logp: float, tpsa: float) -> int:
                 """Calculate Egan rule violations"""
                 violations = 0
                 if logp < -1 or logp > 5.88:
                  violations += 1
                  if tpsa > 131.6:
                   violations += 1
                   return violations

                  def _detect_reactive_groups(
                    self, mol: Chem.Mol) -> List[str]:
                   """Detect reactive functional groups"""
                   reactive_groups = []

    # Check predefined patterns
                   for group_name, smarts in self.reactive_patterns.items():
                    pattern = Chem.MolFromSmarts(
                     smarts)
                    if pattern and mol.HasSubstructMatch(
                      pattern):
                     reactive_groups.append(
                      group_name)

    # Check filter catalog (PAINS, Brenk)
                     if self.filter_catalog.HasMatch(
                       mol):
                      matches = self.filter_catalog.GetMatches(
                       mol)
                      for match in matches:
                       reactive_groups.append(
                        f"Filter_{match.GetDescription()}")

                       # Remove
                       # duplicates
                       return list(
                        set(reactive_groups))

                      def _predict_thermodynamic_stability(
                        self, mol: Chem.Mol, mw: float, logp: float, formal_charge: int) -> float:
                       """Predict thermodynamic stability (0-100)"""
                       score = 80.0  # Base score

    # Molecular weight penalty
                       if mw > 600:
                        score -= (
                         mw - 600) * 0.02
                       elif mw < 100:
                        score -= (
                         100 - mw) * 0.1

    # LogP penalty for extreme values
                        if abs(
                          logp) > 5:
                         score -= abs(
                          logp - 5) * 5

    # Formal charge penalty
                         score -= abs(
                          formal_charge) * 10

    # Ring strain analysis
                         ring_info = mol.GetRingInfo()
                         for ring in ring_info.AtomRings():
                          ring_size = len(
                           ring)
                          # Three-membered
                          # rings
                          # (high
                          # strain)
                          if ring_size == 3:
                          score -= 20
                         # Four-membered
                         # rings
                         # (moderate
                         # strain)
                         elif ring_size == 4:
                         score -= 10
                        # Large
                        # rings
                        # (conformational
                        # instability)
                        elif ring_size > 8:
                        score -= 5

                        return max(
                         0.0, min(100.0, score))

                       def _predict_kinetic_stability(
                         self, mol: Chem.Mol, reactive_groups: List[str], aromatic_rings: int) -> float:
                        """Predict kinetic stability (0-100)"""
                        score = 90.0  # Base score

    # Penalty for reactive groups
                        high_risk_groups = [
                         'aldehyde',
                         'epoxide',
                         'Michael_acceptor',
                         'acid_chloride',
                         'isocyanate',
                         'azide',
                         'diazonium',
                         'peroxide']
                        moderate_risk_groups = [
                         'ketone', 'ester', 'nitrile', 'nitro', 'alkyl_halide']

                        for group in reactive_groups:
                         if any(
                           hrg in group for hrg in high_risk_groups):
                          score -= 25
                         elif any(mrg in group for mrg in moderate_risk_groups):
                          score -= 10
                         else:
                          score -= 5

    # Aromatic stabilization bonus
                          score += min(
                           aromatic_rings * 5, 20)

                          return max(
                           0.0, min(100.0, score))

                         def _predict_metabolic_stability(
                           self, mol: Chem.Mol, logp: float, tpsa: float) -> float:
                          """Predict metabolic stability (0-100)"""
                          score = 70.0  # Base score

    # LogP effect on metabolism
                          if 1 <= logp <= 3:  # Optimal range
                          score += 20
                         elif logp > 5 or logp < 0:
                          score -= 15

    # TPSA effect
                          if 60 <= tpsa <= 90:  # Optimal range
                          score += 10
                         elif tpsa > 140:
                          score -= 20

    # Check for metabolically labile groups
                          labile_patterns = {
                           'ester': '[CX3](=O)[OX2H0][#6]',
                           'amide': '[CX3](=O)[NX3]',
                           'ether': '[OX2]([#6])[#6]',
                           'aromatic_methyl': '[CH3][cX3]'
                          }

                          for group, smarts in labile_patterns.items():
                           pattern = Chem.MolFromSmarts(
                            smarts)
                           if pattern and mol.HasSubstructMatch(
                             pattern):
                            matches = mol.GetSubstructMatches(
                             pattern)
                            score -= len(
                             matches) * 5

                            return max(
                             0.0, min(100.0, score))

                           def _predict_photostability(
                             self, mol: Chem.Mol, aromatic_rings: int) -> float:
                            """Predict photostability (0-100)"""
                            score = 85.0  # Base score

    # Extended conjugation penalty
                            if aromatic_rings > 3:
                             score -= (
                              aromatic_rings - 3) * 10

    # Check for photolabile groups
                             photolabile_patterns = {
                              'nitro_aromatic': '[cX3][NX3](=O)=O',
                              'carbonyl_aromatic': '[cX3][CX3]=O',
                              'aromatic_amine': '[cX3][NX3H2,NX3H1]',
                              'phenol': '[cX3][OX2H]'
                             }

                             for group, smarts in photolabile_patterns.items():
                              pattern = Chem.MolFromSmarts(
                               smarts)
                              if pattern and mol.HasSubstructMatch(
                                pattern):
                               score -= 15

                               return max(
                                0.0, min(100.0, score))

                              def _assess_stability_risk(self, overall_stability: float,
                                    reactive_groups: List[str]) -> StabilityRisk:
                               """Assess overall stability risk"""
                               high_risk_indicators = [
                                'epoxide', 'peroxide', 'azide', 'diazonium', 'acid_chloride']

    # Check for critical reactive groups
                               if any(group in str(
                                 reactive_groups) for group in high_risk_indicators):
                                return StabilityRisk.VERY_HIGH

    # Score-based assessment
                               if overall_stability >= 80:
                                return StabilityRisk.VERY_LOW
                              elif overall_stability >= 65:
                               return StabilityRisk.LOW
                             elif overall_stability >= 50:
                              return StabilityRisk.MODERATE
                            elif overall_stability >= 30:
                             return StabilityRisk.HIGH
                           else:
                            return StabilityRisk.VERY_HIGH

                           def _assess_toxicity_risk(
                             self, mol: Chem.Mol, reactive_groups: List[str]) -> ToxicityLevel:
                            """Assess toxicity risk"""
                            toxic_patterns = {
                             'aromatic_amine': '[cX3][NX3H2,NX3H1]',
                             'nitro_compound': '[NX3](=O)=O',
                             'heavy_metal': '[Fe,Cu,Pb,Hg,Cd,As]',
                             'alkyl_halide': '[CX4][Cl,Br,I]'
                            }

                            toxicity_score = 0

    # Check for toxic functional groups
                            for pattern_name, smarts in toxic_patterns.items():
                             pattern = Chem.MolFromSmarts(
                              smarts)
                             if pattern and mol.HasSubstructMatch(
                               pattern):
                              toxicity_score += 20

    # Reactive group penalties
                              high_tox_groups = [
                               'nitro', 'epoxide', 'alkyl_halide', 'isocyanate']
                              for group in reactive_groups:
                               if any(
                                 htg in group for htg in high_tox_groups):
                                toxicity_score += 15

    # Molecular weight consideration
                                mw = Descriptors.MolWt(
                                 mol)
                                if mw > 800:  # Very large molecules
                                toxicity_score += 10

    # Classify toxicity level
                                if toxicity_score >= 60:
                                 return ToxicityLevel.SEVERE
                               elif toxicity_score >= 40:
                                return ToxicityLevel.HIGH
                              elif toxicity_score >= 20:
                               return ToxicityLevel.MODERATE
                             elif toxicity_score >= 5:
                              return ToxicityLevel.MILD
                            else:
                             return ToxicityLevel.NON_TOXIC

                            def _predict_admet(self, mol: Chem.Mol, mw: float, logp: float,
                                tpsa: float, hb_donors: int, hb_acceptors: int) -> Dict[str, float]:
                             """Predict ADMET properties"""

    # Absorption prediction
                             absorption = 95.0  # Base
                             if tpsa > 140:
                              absorption -= (
                               tpsa - 140) * 0.3
                              if mw > 500:
                               absorption -= (
                                mw - 500) * 0.1
                               if logp < -2 or logp > 5:
                                absorption -= 20

    # Distribution prediction
                                distribution = 80.0  # Base
                                if 1 <= logp <= 3:
                                 distribution += 15
                                elif logp > 5:
                                 distribution -= 25
                                 if tpsa > 90:
                                  distribution -= 10

    # Metabolism prediction
                                  metabolism = 70.0  # Base
                                  if 2 <= logp <= 4:
                                   metabolism += 20
                                   if mw < 300:
                                    metabolism += 10

    # Excretion prediction
                                    excretion = 75.0  # Base
                                    if mw < 300 and tpsa > 75:
                                     excretion += 20
                                     if logp > 4:
                                      excretion -= 15

    # Toxicity prediction
                                      # Base
                                      # (lower
                                      # is
                                      # better)
                                      toxicity = 10.0
                                      if mw > 600:
                                       toxicity += 15
                                       if logp > 5:
                                        toxicity += 20
                                        if hb_donors > 5 or hb_acceptors > 10:
                                         toxicity += 10

                                         return {
                                          'absorption': max(
                                           0, min(
                                            100, absorption)), 'distribution': max(
                                           0, min(
                                            100, distribution)), 'metabolism': max(
                                           0, min(
                                            100, metabolism)), 'excretion': max(
                                           0, min(
                                            100, excretion)), 'toxicity': max(
                                           0, min(
                                            100, toxicity))}

                                        def _generate_recommendations(self, mol: Chem.Mol, stability: float,
                                               reactive_groups: List[str],
                                               lipinski_violations: int) -> Tuple[List[str], List[str]]:
                                         """Generate recommendations and warnings"""
                                         recommendations = []
                                         warnings = []

    # Stability recommendations
                                         if stability < 50:
                                          recommendations.append(
                                           "Consider structural modifications to improve stability")
                                          warnings.append(
                                           " Low overall stability detected")

    # Lipinski violations
                                          if lipinski_violations > 2:
                                           recommendations.append(
                                            "Reduce molecular weight or lipophilicity for better drug-likeness")
                                           warnings.append(
                                            f" {lipinski_violations} Lipinski violations")

    # Reactive group warnings
                                           dangerous_groups = [
                                            'epoxide', 'peroxide', 'azide', 'acid_chloride', 'isocyanate']
                                           for group in reactive_groups:
                                            if any(
                                              dg in group for dg in dangerous_groups):
                                             warnings.append(
                                              f" Highly reactive group detected: {group}")
                                             recommendations.append(
                                              f"Consider replacing {group} with a more stable alternative")

    # Molecular weight recommendations
                                             mw = Descriptors.MolWt(
                                              mol)
                                             if mw > 600:
                                              recommendations.append(
                                               "Consider reducing molecular weight for better bioavailability")

    # LogP recommendations
                                              logp = Crippen.MolLogP(
                                               mol)
                                              if logp > 5:
                                               recommendations.append(
                                                "Reduce lipophilicity by adding polar groups")
                                              elif logp < 0:
                                               recommendations.append(
                                                "Increase lipophilicity for better membrane permeability")

    # Default recommendations if molecule looks good
                                               if not recommendations and stability > 70:
                                                recommendations.append(
                                                 "Molecule shows good stability profile")
                                                recommendations.append(
                                                 "Consider further optimization for specific targets")

                                                return recommendations, warnings

                                               def create_stability_report(self, analysis: StabilityAnalysis,
                                                      output_path: Optional[str] = None) -> str:
                                                """Create comprehensive stability report"""

                                                report = f"""
                                                MOLECULAR STABILITY ANALYSIS REPORT
                                                {'=' * 50}

                                                BASIC PROPERTIES
                                                • Molecular Weight: {analysis.molecular_weight:.2f} Da
                                                • LogP: {analysis.logp:.2f}
                                                • TPSA: {analysis.tpsa:.2f} Ų
                                                • Formal Charge: {analysis.formal_charge:+d}

                                                🏗 STRUCTURAL FEATURES
                                                • Total Rings: {analysis.num_rings}
                                                • Aromatic Rings: {analysis.aromatic_rings}
                                                • Rotatable Bonds: {analysis.rotatable_bonds}
                                                • H-Bond Donors: {analysis.hb_donors}
                                                • H-Bond Acceptors: {analysis.hb_acceptors}

                                                📋 DRUG-LIKENESS ASSESSMENT
                                                • Lipinski Violations: {analysis.lipinski_violations}/4
                                                • Veber Violations: {analysis.veber_violations}/2
                                                • Egan Violations: {analysis.egan_violations}/2

                                                STABILITY ANALYSIS
                                                • Thermodynamic Stability: {analysis.thermodynamic_stability:.1f}/100
                                                • Kinetic Stability: {analysis.kinetic_stability:.1f}/100
                                                • Metabolic Stability: {analysis.metabolic_stability:.1f}/100
                                                • Photostability: {analysis.photostability:.1f}/100
                                                • Overall Stability: {analysis.overall_stability:.1f}/100

                                                RISK ASSESSMENT
                                                • Stability Risk: {analysis.stability_risk.value[0]}
                                                • Toxicity Risk: {analysis.toxicity_risk.value[0]}

                                                ADMET PREDICTIONS
                                                • Absorption: {analysis.absorption:.1f}/100
                                                • Distribution: {analysis.distribution:.1f}/100
                                                • Metabolism: {analysis.metabolism:.1f}/100
                                                • Excretion: {analysis.excretion:.1f}/100
                                                • Toxicity Score: {analysis.toxicity:.1f}/100

                                                🚨 REACTIVE GROUPS DETECTED
                                                {chr(10).join(f"• {group}" for group in analysis.reactive_groups) if analysis.reactive_groups else "• No reactive groups detected"}

                                                RECOMMENDATIONS
                                                {chr(10).join(f"• {rec}" for rec in analysis.recommendations)}

                                                WARNINGS
                                                {chr(10).join(
                                                 f"• {warn}" for warn in analysis.warnings) if analysis.warnings else "• No warnings"}

                                                {'=' * 50}
                                                Report generated by Advanced Stability Analyzer
                                                """

                                                if output_path:
                                                 with open(output_path, 'w') as f:
                                                  f.write(
                                                   report)
                                                  print(
                                                   f" Report saved to: {output_path}")

                                                  return report

                                                 def batch_analyze(self, molecules: List[Union[str, Chem.Mol]],
                                                     output_dir: str = "stability_analysis") -> pd.DataFrame:
                                                  """Analyze multiple molecules in batch"""
                                                  output_path = Path(
                                                   output_dir)
                                                  output_path.mkdir(
                                                   exist_ok=True, parents=True)

                                                  results = []

                                                  print(
                                                   f" Analyzing {len(molecules)} molecules...")
                                                  from tqdm import tqdm

                                                  for i, mol_input in tqdm(enumerate(
                                                    molecules), total=len(molecules)):
                                                   try:
                                                    analysis = self.analyze_molecule(
                                                     mol_input)

    # Convert to dictionary for DataFrame
                                                    result = asdict(
                                                     analysis)
                                                    result[
                                                     'molecule_id'] = f"mol_{i}"
                                                    result['smiles'] = Chem.MolToSmiles(mol_input) if isinstance(
                                                     mol_input, Chem.Mol) else mol_input

    # Convert enum values to strings
                                                    result[
                                                     'stability_risk'] = analysis.stability_risk.value[0]
                                                    result[
                                                     'toxicity_risk'] = analysis.toxicity_risk.value[0]
                                                    result['reactive_groups'] = ', '.join(
                                                     analysis.reactive_groups)
                                                    result['recommendations'] = ' | '.join(
                                                     analysis.recommendations)
                                                    result['warnings'] = ' | '.join(
                                                     analysis.warnings)

                                                    results.append(
                                                     result)

    # Save individual report
                                                    report = self.create_stability_report(
                                                     analysis)
                                                    report_file = output_path / \
                                                     f"stability_report_mol_{i}.txt"
                                                    with open(report_file, 'w') as f:
                                                     f.write(
                                                      report)

                                                    except Exception as e:
                                                     print(
                                                      f" Error analyzing molecule {i}: {e}")
                                                     continue

    # Create summary DataFrame
                                                    df = pd.DataFrame(
                                                     results)

    # Save summary
                                                    summary_file = output_path / "stability_summary.csv"
                                                    df.to_csv(
                                                     summary_file, index=False)

                                                    print(
                                                     f" Batch analysis complete. Results saved to: {output_path}")
                                                    return df

                                                   def main():
                                                    """Main execution for testing"""
                                                    print(
                                                     """
                                                    ═══════════════════════════════════════════════════════════════
                                                    ADVANCED MOLECULAR STABILITY ANALYSIS SYSTEM

                                                    Features:
                                                     • Comprehensive stability prediction
                                                     • Reactive group detection
                                                     • ADMET property estimation
                                                     • Toxicity risk assessment
                                                     • Drug-likeness evaluation

                                                     Ready for molecular analysis!
                                                     ═══════════════════════════════════════════════════════════════
                                                     """)

    # Example usage
                                                    analyzer = AdvancedStabilityAnalyzer()

    # Test molecules
                                                    test_molecules = [
                                                     "CCO",  # Ethanol
                                                     # Aspirin
                                                     "CC(=O)OC1=CC=CC=C1C(=O)O",
                                                     # Caffeine
                                                     "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                                                     # Ibuprofen
                                                     "CC1=CC=C(C=C1)C(C)C(=O)O",
                                                    ]

                                                    print(
                                                     "\n Analyzing test molecules...")
                                                    for i, smiles in enumerate(
                                                      test_molecules):
                                                     print(
                                                      f"\n--- Molecule {i + 1}: {smiles} ---")
                                                     try:
                                                      analysis = analyzer.analyze_molecule(
                                                       smiles)
                                                      print(
                                                       f"Overall Stability: {analysis.overall_stability:.1f}/100")
                                                      print(
                                                       f"Stability Risk: {analysis.stability_risk.value[0]}")
                                                      print(
                                                       f"Toxicity Risk: {analysis.toxicity_risk.value[0]}")
                                                      if analysis.warnings:
                                                        print(
                                                         f"Warnings: {', '.join(analysis.warnings)}")
                                                       except Exception as e:
                                                        print(
                                                         f"Error: {e}")

                                                        if __name__ == "__main__":
                                                         main()
