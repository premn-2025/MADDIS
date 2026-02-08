#!/usr/bin/env python3
"""
Complete Molecular Dataset Pipeline for Fine-Tuning

ONE-CLICK solution to download and prepare real drug molecules for AI training.
Optimized for RTX 4060 + i7 14th gen.

What this script does:
    1. Downloads 60k real drug molecules from ChEMBL (SMILES format)
    2. Converts SMILES to 2D images locally using RDKit (NO web scraping!)
    3. Generates 3D conformers for docking/GNN training
    4. Prepares datasets for Vision, Multimodal, and Graph models
    5. Creates train/val splits ready for fine-tuning

    USAGE:
        python examples/build_molecular_dataset.py

        Author: AI Drug Discovery Pipeline
        """

from src.data.fine_tuning_prep import FineTuningDataPreparer, FineTuningConfig
from src.data.dataset_builder import MolecularDatasetBuilder, DatasetConfig
import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_banner():
    """Print welcome banner"""
    print("""
            ═══════════════════════════════════════════════════════════════════════════════════════════════
            MOLECULAR DATASET BUILDER FOR AI DRUG DISCOVERY

            Downloads REAL molecules from ChEMBL (not random images!)
            Converts SMILES → 2D/3D representations locally
            Optimized for RTX 4060 + i7 14th gen
            Ready for Vision/Multimodal/Graph model fine-tuning
            Production-grade chemical intelligence
            ═══════════════════════════════════════════════════════════════════════════════════════════════
            """)

    def check_requirements():
        """Check that required packages are installed"""
        print(" Checking requirements...")

        required = ["rdkit", "torch", "pandas", "requests", "tqdm", "PIL"]
        missing = []

        for package in required:
            try:
                    if package == "rdkit":
                        import rdkit
                    elif package == "torch":
                        import torch
                    elif package == "PIL":
                        from PIL import Image
                    else:
                        __import__(package)
                        print(f" {package}")
                    except ImportError:
                        missing.append(package)
                        print(f" {package} (missing)")

                        if missing:
                            print(f"\n Missing packages: {', '.join(missing)}")
                            print("\n Install with:")
                            print(
                                " pip install rdkit-pypi torch torchvision pandas requests tqdm pillow")
                            print(" # OR for conda:")
                            print(
                                " conda install -c conda-forge rdkit pytorch pandas requests tqdm pillow")
                            return False

                        print(" All requirements satisfied!")
                        return True

                    def get_user_preferences():
                        """Get user preferences for dataset generation"""
                        print("\n Dataset Configuration")
                        print("═" * 50)

# Number of molecules
                        while True:
                            try:
                                num_molecules = input(
                                    " Number of molecules to download (default: 60000): ").strip()
                                if not num_molecules:
                                    num_molecules = 60000
                                else:
                                    num_molecules = int(num_molecules)

                                    if num_molecules > 0:
                                        break
                                    print(" Please enter a positive number")
                                except ValueError:
                                    print(" Please enter a valid number")

# Data source
                                    print("\n Data source options:")
                                    print(
                                        " 1. ChEMBL (recommended) - Curated bioactive molecules")
                                    print(
                                        " 2. ZINC - Purchasable drug-like compounds")

                                    while True:
                                        choice = input(
                                            "Choose source (1/2, default: 1): ").strip()
                                        if not choice or choice == "1":
                                            source = "chembl"
                                            break
                                    elif choice == "2":
                                        source = "zinc"
                                        break
                                    print(" Please choose 1 or 2")

# Output directory
                                    output_dir = input(
                                        "\n📁 Output directory (default: drug_dataset): ").strip()
                                    if not output_dir:
                                        output_dir = "drug_dataset"

# Generate 3D structures
                                        print(
                                            "\n Generate 3D molecular structures?")
                                        print(
                                            " Recommended for docking and advanced GNN training")
                                        print(" Takes more time and disk space")

                                        while True:
                                            choice = input(
                                                "Generate 3D (y/n, default: y): ").strip().lower()
                                            if not choice or choice in [
             "y", "yes"]:
                                                generate_3d = True
                                                break
                                        elif choice in ["n", "no"]:
                                            generate_3d = False
                                            break
                                        print(" Please enter y or n")

                                        return {
                                            "num_molecules": num_molecules,
                                            "source": source,
                                            "output_dir": output_dir,
                                            "generate_3d": generate_3d
                                        }

                                    def estimate_requirements(prefs):
                                        """Estimate time and disk space requirements"""
                                        num_mols = prefs["num_molecules"]

# Time estimates (rough)
                                        download_time = num_mols * 0.01 / 60  # seconds to minutes
                                        image_time = num_mols * 0.1 / 60  # seconds to minutes
                                        conformer_time = num_mols * 2 / \
                                            60 if prefs["generate_3d"] else 0  # seconds to minutes

                                        total_time = download_time + image_time + conformer_time

# Disk space estimates
                                        smiles_mb = num_mols * 0.1 / 1024  # KB to MB
                                        images_mb = num_mols * 50 / 1024  # KB to MB
                                        conformers_mb = num_mols * 5 / \
                                            1024 if prefs["generate_3d"] else 0  # KB to MB

                                        total_mb = smiles_mb + images_mb + conformers_mb

                                        print(f"\n Estimated Requirements:")
                                        print(
                                            f" ⏱ Time: ~{
                                                total_time:.1f} minutes")
                                        print(
                                            f" Disk: ~{
                                                total_mb:.1f} MB ({
                                                total_mb /
                                                1024:.1f} GB)")
                                        print(f" Molecules: {num_mols:,}")
                                        print(f" 2D Images: {num_mols:,}")
                                        if prefs["generate_3d"]:
                                            print(
                                                f" 3D Structures: {
             num_mols:,}")

                                            def build_molecular_dataset(prefs):
                                                """Build the molecular dataset"""
                                                print(
             f"\n🏗 Building Molecular Dataset")
                                                print("═" * 50)

# Configure dataset builder
                                                config = DatasetConfig(
             source=prefs["source"],
             total_molecules=prefs["num_molecules"],
             output_dir=prefs["output_dir"],
             generate_2d_images=True,
             generate_3d_conformers=prefs["generate_3d"],
             image_size=(300, 300),
             filter_drug_like=True,
             batch_size=1000,
             delay_between_requests=0.1
                                                )

# Build dataset
                                                builder = MolecularDatasetBuilder(
             config)

                                                print(
             f" Downloading from {
              prefs['source'].upper()}...")
                                                start_time = time.time()

                                                results = builder.build_dataset()

                                                end_time = time.time()
                                                duration = (
             end_time - start_time) / 60  # minutes

                                                if results["success"]:
             print(
              f"\n Dataset Build Complete!")
             print(
              f" ⏱ Time: {duration:.1f} minutes")
             print(
              f" Molecules: {
               results['total_molecules']:,}")
             print(
              f" Images: {
               results['images_generated']:,}")
             print(
              f" 3D Structures: {
               results['conformers_generated']:,}")
             return results
                                            else:
                                                print(
             f"\n Dataset build failed: {
              results['error']}")
                                                return None

                                            def prepare_fine_tuning_data(
             dataset_results, prefs):
                                                """Prepare data for fine-tuning various model types"""
                                                if not dataset_results:
             return None

                                                print(
             f"\n Preparing Fine-Tuning Data")
                                                print("═" * 50)

# Configure fine-tuning prep
                                                config = FineTuningConfig(
             dataset_dir=prefs["output_dir"],
             output_dir=f"{
              prefs['output_dir']}/fine_tuning_ready",
             prepare_vision=True,  # For ViT, ResNet, EfficientNet
             prepare_multimodal=True,  # For CLIP, BLIP
             prepare_graph=True,  # For GNNs
             image_size=224,  # Standard for most models
             augmentation_strength="medium",
             batch_size=32,  # Optimal for RTX 4060
             num_workers=4,  # Good for i7 14th gen
             property_targets=[
              "molecular_weight", "logp", "hba", "hbd", "tpsa"]
                                                )

# Prepare datasets
                                                preparer = FineTuningDataPreparer(
             config)
                                                results = preparer.prepare_all_datasets()

                                                print(
             f"\n Fine-tuning datasets ready!")
                                                print(
             f" Vision models: {
              '' if 'vision' in results['datasets'] else ''}")
                                                print(
             f" Multimodal models: {
              '' if 'multimodal' in results['datasets'] else ''}")
                                                print(
             f" Graph models: {
              '' if 'graph' in results['datasets'] else ''}")
                                                print(
             f" 🎮 Device: {
              results['device']}")

                                                return results

                                            def show_next_steps(
             dataset_results, finetuning_results, prefs):
                                                """Show user what they can do next"""
                                                print(f"\n What's Next?")
                                                print("═" * 50)

                                                print(
             f"📁 Your dataset is ready in: {
              prefs['output_dir']}/")
                                                print("\n Folder structure:")
                                                print(
             f" {
              prefs['output_dir']}/smiles/molecules.smi")
                                                print(
             f" {
              prefs['output_dir']}/images_2d/ (60k PNG files)")
                                                if prefs["generate_3d"]:
             print(
              f" {
               prefs['output_dir']}/sdf_3d/ (60k SDF files)")
             print(
              f" {
               prefs['output_dir']}/metadata/molecules_metadata.csv")
             print(
              f" {
               prefs['output_dir']}/fine_tuning_ready/ (PyTorch datasets)")

             print(
              f"\n Ready for fine-tuning:")

             print(
              f" 1⃣ Vision Models (ResNet, ViT, EfficientNet):")
             print(
              f" from fine_tuning_ready.vision import train_loader, val_loader")

             print(
              f" 2⃣ Multimodal Models (CLIP, BLIP):")
             print(
              f" from fine_tuning_ready.multimodal import train_loader, val_loader")

             print(
              f" 3⃣ Graph Neural Networks (GCN, GAT, GraphSAINT):")
             print(
              f" from fine_tuning_ready.graph import train_loader, val_loader")

             print(
              f"\n💪 RTX 4060 Optimization Tips:")
             print(
              f" • Batch size: 32 (already set)")
             print(
              f" • Mixed precision: Use torch.cuda.amp for 2x speedup")
             print(
              f" • Gradient accumulation: 2-4 steps for larger effective batch")
             print(
              f" • Memory: Monitor with nvidia-smi")

             print(
              f"\n📚 Example fine-tuning scripts available in:")
             print(
              f" examples/fine_tune_vision.py")
             print(
              f" examples/fine_tune_multimodal.py")
             print(
              f" examples/fine_tune_graph.py")

             def main():
              """Main pipeline execution"""
              print_banner()

# Check requirements
              if not check_requirements():
               print(
                "\n Please install missing packages first!")
               return 1

# Get user preferences
              prefs = get_user_preferences()

# Show estimates
              estimate_requirements(
               prefs)

# Confirm
              print(
               f"\n🚦 Ready to proceed?")
              confirm = input(
               "Start building dataset? (y/n): ").strip().lower()
              if confirm not in [
                "y", "yes"]:
               print(
                "👋 Cancelled by user")
               return 0

# Build molecular dataset
              dataset_results = build_molecular_dataset(
               prefs)
              if not dataset_results:
               return 1

# Prepare fine-tuning data
              finetuning_results = prepare_fine_tuning_data(
               dataset_results, prefs)

# Show next steps
              show_next_steps(
               dataset_results, finetuning_results, prefs)

              print(
               f"\n Complete! Your molecular dataset is ready for AI training.")
              return 0

             if __name__ == "__main__":
              exit_code = main()
              sys.exit(exit_code)
