"""
Multi-Agent AI Drug Discovery Pipeline

Complete implementation of AI-driven drug discovery methodology including:
    - Data collection from major databases (ChEMBL, PubChem, ZINC, BindingDB)
    - ML-based property prediction with multiple architectures
    - Generative AI for de novo molecule design
    - Structure-based docking and scoring
    - LLM orchestration for analysis and decision making
    - 3D molecular visualization
    - Iterative optimization loops
    """

# Import all major components
from .data import *
from .preprocessing import *
from .models import *
from .generation import *
from .docking import *
from .orchestration import *
from .visualization import *
from .optimization import *

__version__ = "1.0.0"
__author__ = "AI Drug Discovery Team"

__all__ = [
    # Data collection
    'DataManager', 'ChEMBLCollector', 'PubChemCollector', 'ZINCCollector',
    'BindingDBCollector', 'PDBCollector', 'DataUtils', 'DatasetBuilder',

    # Preprocessing
    'MoleculeRepresentation', 'RDKitPreprocessor', 'SMILESTokenizer',
    'BatchPreprocessor', 'DrugFeatureEngineer', 'ProteinFeatureEngineer',

    # ML Models
    'PropertyPredictor', 'RandomForestPredictor', 'DeepPredictor',
    'GNNPredictor', 'BasicGCN', 'GraphAttentionNetwork',

    # Generation
    'MolecularGenerator', 'VAEGenerator', 'GeneticAlgorithm',
    'FragmentBasedGenerator',

    # Docking
    'DockingPipeline', 'MolecularDocking', 'BindingSiteAnalyzer',

    # Orchestration
    'DrugDiscoveryOrchestrator', 'OpenAIProvider', 'AnthropicProvider',

    # Visualization
    'MolecularVisualizationSuite', 'PyMOLVisualizer', 'PlotlyVisualizer',

    # Optimization
    'DrugDiscoveryOptimizer', 'OptimizationConfig'
]
