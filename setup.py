#!/usr/bin/env python3
"""
Installation and Setup Script for Multi-Agent AI Drug Discovery Pipeline

This script sets up the complete development environment, creates necessary
directories, initializes configurations, and validates all components.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import json


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def check_python_version():
    """Ensure Python 3.7+ is being used."""
    print_header("Checking Python Version")

    if sys.version_info < (3, 7):
        print(" Error: Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)

    print(
        f" Python "
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro} is supported")


def install_requirements():
    """Install required Python packages."""
    print_header("Installing Python Dependencies")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" All Python dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f" Error installing dependencies: {e}")
        print(" Try running: pip install --user -r requirements.txt")
        return False

    return True


def create_directories():
    """Create necessary project directories."""
    print_header("Creating Project Structure")

    directories = [
        "src",
        "src/data",
        "src/preprocessing",
        "src/models",
        "src/generation",
        "src/docking",
        "src/orchestration",
        "src/visualization",
        "src/optimization",
        "outputs",
        "outputs/iterations",
        "outputs/visualizations",
        "outputs/logs",
        "data/raw",
        "data/processed",
        "data/cache",
        "models/trained",
        "models/checkpoints",
        "docs/tutorials",
        "tests/unit",
        "tests/integration"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f" Created: {directory}/")


def setup_environment():
    """Create environment configuration template."""
    print_header("Setting Up Environment Configuration")

    env_template = """# Multi-Agent AI Drug Discovery Pipeline - Environment Configuration

# =============================================================================
# LLM API KEYS - Configure your API access
# =============================================================================

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_organization_id_here # Optional

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =============================================================================
# Database Configuration
# =============================================================================

# ChEMBL Configuration
CHEMBL_BASE_URL=https://www.ebi.ac.uk/chembl/api/data
CHEMBL_CACHE_DAYS=7

# PubChem Configuration
PUBCHEM_BASE_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug
PUBCHEM_CACHE_DAYS=7

# =============================================================================
# Computational Resources
# =============================================================================

# GPU Configuration
CUDA_VISIBLE_DEVICES=0 # Set to your GPU ID, or remove for CPU-only

# Parallel Processing
N_JOBS=4 # Number of CPU cores to use

# Memory Limits
MAX_MEMORY_GB=8 # Maximum memory usage

# =============================================================================
# External Tools (Optional - will use fallbacks if not available)
# =============================================================================

# PyMOL Installation Path (for 3D visualization)
PYMOL_PATH=/path/to/pymol # Update with your PyMOL installation

# AutoDock Vina Path (for molecular docking)
VINA_PATH=/path/to/vina # Update with your Vina installation

# =============================================================================
# Development Configuration
# =============================================================================

# Logging Level
LOG_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR

# Debug Mode
DEBUG_MODE=False

# Cache Settings
ENABLE_CACHE=True
CACHE_TTL_HOURS=24
"""

    env_file = ".env"
    if not Path(env_file).exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print(f" Created environment template: {env_file}")
        print(" Please edit .env file with your API keys and configuration")
    else:
        print(f" Environment file already exists: {env_file}")


def validate_imports():
    """Validate that key dependencies can be imported."""
    print_header("Validating Package Imports")

    required_packages = [
        ("numpy", "Scientific computing"),
        ("pandas", "Data manipulation"),
        ("scikit-learn", "Machine learning"),
        ("torch", "Deep learning"),
        ("rdkit", "Molecular handling"),
        ("requests", "HTTP requests"),
        ("yaml", "Configuration files")
    ]

    optional_packages = [
        ("torch_geometric", "Graph neural networks"),
        ("pymol", "3D molecular visualization"),
        ("plotly", "Interactive plotting"),
        ("openai", "OpenAI API"),
        ("anthropic", "Anthropic API")
    ]

    all_good = True

    print("\n Required packages:")
    for package, description in required_packages:
        try:
            __import__(package)
            print(f" {package:<15} - {description}")
        except ImportError:
            print(f" {package:<15} - {description} (MISSING)")
            all_good = False

    print("\n Optional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f" {package:<15} - {description}")
        except ImportError:
            print(f" {package:<15} - {description} (missing - fallback available)")

    return all_good


def create_quick_test():
    """Create a quick validation test script."""
    print_header("Creating Validation Test")

    test_script = '''#!/usr/bin/env python3
"""
Quick validation test for Multi-Agent AI Drug Discovery Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that core modules can be imported."""
    print(" Testing core module imports...")

    try:
        from src.data import DataManager
        print(" Data module imported successfully")

        from src.preprocessing import RDKitPreprocessor
        print(" Preprocessing module imported successfully")

        from src.models import PropertyPredictor
        print(" Models module imported successfully")

        from src.generation import MolecularGenerator
        print(" Generation module imported successfully")

        from src.docking import MolecularDocking
        print(" Docking module imported successfully")

        from src.orchestration import DrugDiscoveryOrchestrator
        print(" Orchestration module imported successfully")

        from src.visualization import MolecularVisualizationSuite
        print(" Visualization module imported successfully")

        from src.optimization import DrugDiscoveryOptimizer
        print(" Optimization module imported successfully")

        return True

    except Exception as e:
        print(f" Import error: {e}")
        return False


def test_simple_molecule_processing():
    """Test basic molecule processing functionality."""
    print("\\n Testing molecule processing...")

    try:
        from src.preprocessing.molecular import RDKitPreprocessor

        processor = RDKitPreprocessor()

        # Test with aspirin SMILES
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = processor.process_smiles(aspirin_smiles)

        if result and 'fingerprint' in result:
            print(" Molecule processing works correctly")
            return True
        else:
            print(" Molecule processing failed")
            return False

    except Exception as e:
        print(f" Molecule processing error: {e}")
        return False


def main():
    """Run validation tests."""
    print(" Multi-Agent AI Drug Discovery Pipeline - Validation Test")
    print("="*60)

    all_tests_passed = True

    # Test imports
    if not test_imports():
        all_tests_passed = False

    # Test molecule processing
    if not test_simple_molecule_processing():
        all_tests_passed = False

    print("\\n" + "="*60)
    if all_tests_passed:
        print(" All validation tests passed! Pipeline is ready to use.")
        print("\\n Next steps:")
        print(" 1. Configure API keys in .env file")
        print(" 2. Run: python examples/example_pipeline.py")
    else:
        print(" Some validation tests failed. Check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    test_file = "tests/validate_setup.py"
    Path("tests").mkdir(exist_ok=True)

    with open(test_file, 'w') as f:
        f.write(test_script)

    # Make executable
    os.chmod(test_file, 0o755)

    print(f" Created validation test: {test_file}")


def main():
    """Main setup function."""
    print(" Multi-Agent AI Drug Discovery Pipeline Setup")
    print("=" * 60)
    print("This script will set up your complete development environment.")
    print("=" * 60)

    # Check Python version
    check_python_version()

    # Create directories
    create_directories()

    # Setup environment
    setup_environment()

    # Install requirements
    if not install_requirements():
        print("\n Setup failed during dependency installation.")
        print(" Please check the error messages above and try again.")
        return 1

    # Validate imports
    if not validate_imports():
        print("\n Some required packages are missing.")
        print(" The pipeline may not work correctly without them.")

    # Create test script
    create_quick_test()

    # Final instructions
    print_header("Setup Complete!")

    print(" Your AI drug discovery pipeline is now set up!")
    print("\n📋 Next steps:")
    print(" 1. Edit .env file with your API keys")
    print(" 2. Run validation: python tests/validate_setup.py")
    print(" 3. Try the example: python examples/example_pipeline.py")
    print(" 4. Read the guide: docs/IMPLEMENTATION_GUIDE.md")

    print("\n Key files created:")
    print(" .env - Configuration template")
    print(" outputs/ - Results directory")
    print(" data/ - Data storage")
    print(" tests/validate_setup.py - Validation test")

    print("\n Pro tips:")
    print(" • Configure .env with your OpenAI/Anthropic API keys")
    print(" • Install PyMOL for advanced 3D visualization")
    print(" • Install AutoDock Vina for molecular docking")
    print(" • Use GPU for faster neural network training")

    return 0


if __name__ == "__main__":
    sys.exit(main())