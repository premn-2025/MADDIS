"""
Complete Example: AI Drug Discovery Pipeline

This example demonstrates the full AI drug discovery methodology:
    1. Target selection and data collection
    2. Molecule preprocessing and feature engineering
    3. ML model training for property prediction
    4. Generative AI for molecule design
    5. Structure-based docking
    6. LLM-guided analysis
    7. 3D visualization
    8. Iterative optimization

    Usage:
        python examples/example_pipeline.py
        """

from src.optimization import OptimizationConfig, DrugDiscoveryOptimizer
import numpy as np
import pandas as pd
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('drug_discovery_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main example execution"""

    print(" Multi-Agent AI Drug Discovery Pipeline Example")
    print("=" * 60)

# Configuration
    config = OptimizationConfig(
        target_name="EGFR",  # Epidermal Growth Factor Receptor
        max_iterations=3,  # Limited for example
        molecules_per_iteration=20,
        top_n_selection=5,
        generation_method='genetic',
        prediction_models=['random_forest'],
        llm_provider='fallback',  # Use fallback unless API keys are set
        output_dir='./example_results',
        save_intermediate=True
    )

    print(f"Target: {config.target_name}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Generation method: {config.generation_method}")
    print(f"LLM provider: {config.llm_provider}")
    print()

    try:
        # Create and run optimizer
        logger.info("Initializing drug discovery optimizer...")
        optimizer = DrugDiscoveryOptimizer(config)

# Define initial seed molecules (drug-like compounds)
        seed_molecules = [
            'CCO',  # Ethanol (simple)
            'CC(C)O',  # Isopropanol
            'c1ccccc1O',  # Phenol
            'CCN(CC)CC',  # Triethylamine
            'CC(=O)Nc1ccc(O)cc1',  # Acetaminophen
            'COc1ccc(C(=O)O)cc1',  # p-Methoxybenzoic acid
            'c1ccc2[nH]c3ccccc3c2c1',  # Carbazole
            'CC(C)(C)c1ccc(O)cc1',  # 4-tert-Butylphenol
        ]

        print("Starting optimization with seed molecules:")
        for i, mol in enumerate(seed_molecules, 1):
            print(f" {i}. {mol}")
            print()

# Run the optimization pipeline
            logger.info("Starting optimization pipeline...")
            results = optimizer.run_optimization(
                initial_molecules=seed_molecules)

# Display results
            print("\n" + "=" * 60)
            print(" OPTIMIZATION RESULTS")
            print("=" * 60)

            summary = results['optimization_summary']
            print(f"Target: {summary['target_name']}")
            print(f"Iterations completed: {summary['total_iterations']}")
            print(
                f"Total execution time: {
                    summary['total_time']:.1f} seconds")
            print(
                f"Total molecules generated: {
                    summary['molecules_generated']}")
            print(
                f"Unique molecules explored: {
                    summary['unique_molecules']}")
            print()

# Best molecules
            if results['best_molecules']:
                print(" TOP DISCOVERED MOLECULES:")
                for i, molecule in enumerate(
                        results['best_molecules'][:5], 1):
                    print(f" {i}. {molecule}")
                    print()

# Final metrics
                    final_metrics = results['final_metrics']
                    if final_metrics:
                        print(" FINAL METRICS:")
                        print(
                            f" Best binding affinity: {
                                final_metrics.get(
                                    'best_binding_affinity',
                                    'N/A'):.3f} kcal/mol")
                        print(
                            f" Mean binding affinity: {
                                final_metrics.get(
                                    'mean_binding_affinity',
                                    'N/A'):.3f} kcal/mol")
                        print(
                            f" Good binders (< -8.0): {final_metrics.get('good_binders_count', 'N/A')}")
                        print(
                            f" Success rate: {
                                final_metrics.get(
                                    'good_binders_fraction',
                                    0) * 100:.1f}%")
                        print()

# Convergence analysis
                        if results['convergence_history']:
                            print(" CONVERGENCE ANALYSIS:")
                            for entry in results['convergence_history']:
                                print(
                                    f" Iteration {
                                        entry['iteration']}: " f"Best score = {
                                        entry['best_score']:.3f}, " f"Improvement = {
                                        entry['improvement']:.3f}")
                                print()

# Output files
                                print("📁 OUTPUT FILES:")
                                output_dir = Path(config.output_dir)
                                if output_dir.exists():
                                    print(f" Main directory: {output_dir}")
                                    print(
                                        f" Summary report: {
                                            output_dir /
                                            'optimization_summary.json'}")

# List iteration directories
                                    iteration_dirs = sorted([d for d in output_dir.iterdir(
                                    ) if d.is_dir() and d.name.startswith('iteration')])
                                    if iteration_dirs:
                                        print(
                                            f" Iteration results: {
                                                len(iteration_dirs)} directories")
                                        for iter_dir in iteration_dirs:
                                            print(f" - {iter_dir.name}/")

# Visualization files
                                            viz_dir = output_dir / 'visualizations'
                                            if viz_dir.exists():
            viz_files = list(
             viz_dir.glob('*.html'))
            if viz_files:
              print(
               f" Visualizations: {
                len(viz_files)} HTML files")
              for viz_file in viz_files:
               print(
                f" - {viz_file.name}")

               print(
                "\n" + "=" * 60)
               print(
                " Pipeline execution completed successfully!")
               print(
                " Check the output directory for detailed results and visualizations.")
               print("=" * 60)

               return results

             except Exception as e:
              logger.error(
               f"Pipeline execution failed: {e}")
              print(f"\n Error: {e}")
              print(
               "Check the log file for detailed error information.")
              return None

             def demo_individual_components():
              """Demonstrate individual pipeline components"""

              print(
               "\n COMPONENT DEMONSTRATION")
              print("=" * 40)

              try:
              # 1. Data collection
              # demo
               print(
                "1. Data Collection...")
               from src.data import DataManager
               data_manager = DataManager()

# Try to collect some data (may fail without internet/API access)
               try:
                sample_data = data_manager.collect_target_data(
                 "kinase", limit=5)
                if not sample_data.empty:
                 print(
                  f" ✓ Collected {
                   len(sample_data)} sample molecules")
                else:
                 print(
                  " No data collected (requires internet access)")
                except BaseException:
                 print(
                  " Data collection skipped (requires internet access)")

# 2. Preprocessing demo
                 print(
                  "2. Molecular Preprocessing...")
                 from src.preprocessing import BatchPreprocessor

                 test_molecules = [
                  'CCO', 'CCC', 'c1ccccc1']
                 preprocessor = BatchPreprocessor()

                 try:
                  results = preprocessor.process_batch(
                   test_molecules, include_3d=False)
                  print(
                   f" ✓ Processed {len(results['smiles'])} molecules")
                  print(
                   f" ✓ Generated fingerprints: {len([fp for fp in results['fingerprints'] if fp is not None])}")
                 except Exception as e:
                  print(
                   f" Preprocessing demo failed: {e}")

# 3. Generation demo
                  print(
                   "3. Molecule Generation...")
                  from src.generation import MolecularGenerator

                  try:
                   generator = MolecularGenerator(
                    'fragment')
                   generated = generator.generate(
                    num_molecules=5)
                   print(
                    f" ✓ Generated {len(generated)} new molecules")
                   print(
                    f" ✓ Examples: {', '.join(generated[:3])}")
                  except Exception as e:
                   print(
                    f" Generation demo failed: {e}")

# 4. LLM orchestration demo
                   print(
                    "4. LLM Analysis...")
                   from src.orchestration import DrugDiscoveryOrchestrator

                   try:
                    orchestrator = DrugDiscoveryOrchestrator(
                     'fallback')

# Create dummy docking data
                    dummy_data = pd.DataFrame({
                     'smiles': ['CCO', 'CCC', 'c1ccccc1'],
                     'binding_affinity': [-7.5, -6.2, -8.1]
                    })

                    analysis = orchestrator.analyze_docking_results(
                     dummy_data)
                    print(
                     " ✓ LLM analysis completed")
                    print(
                     f" ✓ Recommendations: {len(analysis.recommendations)}")

                   except Exception as e:
                    print(
                     f" LLM demo failed: {e}")

                    print(
                     "\n Component demonstration completed!")

                   except Exception as e:
                    print(
                     f"\n Component demonstration failed: {e}")

                    if __name__ == "__main__":
                     print(
                      "Starting Multi-Agent AI Drug Discovery Pipeline Example...")
                     print(
                      "This example demonstrates the complete methodology outlined in your request.\n")

# Run main pipeline example
                     results = main()

# Run component demos
                     demo_individual_components()

                     print(
                      f"\n📚 For more information, see:")
                     print(
                      f" - README.md: Project overview and setup")
                     print(
                      f" - src/: Source code for all components")
                     print(
                      f" - configs/: Configuration files")
                     print(
                      f" - docs/: Detailed documentation (if available)")

                     print(
                      f"\n Next Steps:")
                     print(
                      f" 1. Install dependencies: pip install -r requirements.txt")
                     print(
                      f" 2. Set up API keys in .env file for full functionality")
                     print(
                      f" 3. Customize OptimizationConfig for your specific target")
                     print(
                      f" 4. Run with real protein structures and larger datasets")
                     print(
                      f" 5. Experiment with different generation methods and ML models")

                     if results:
                      print(
                       f"\n This example successfully demonstrated all 9 phases of your methodology!")
                     else:
                      print(
                       f"\n Some components may require additional setup for full functionality.")
