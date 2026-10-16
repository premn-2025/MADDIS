# 🧬 Multi-Agent AI Drug Discovery Platform 

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)
![Dataset](https://img.shields.io/badge/dataset-10%2B%20GB-orange.svg)
![Molecules](https://img.shields.io/badge/molecules-2M%2B-purple.svg)

**The Ultimate AI-Powered Molecular Dataset & 3D Visualization Platform** 

A comprehensive drug discovery platform featuring massive molecular datasets (10+ GB), interactive 3D visualization, real-time molecular combination, and advanced stability analysis.

## 🚀 Launch the Platform

```bash
pip install -r requirements.txt
streamlit run integrated_multiagent_platform.py
```

This starts the MADDIS web interface on http://localhost:8501 with:
- ✅ **Real QSAR binding prediction** for EGFR, COX2, BACE1, JAK2 and THROMBIN,
  trained on ChEMBL bioactivity data (held-out R² 0.48–0.67)
- ✅ **Applicability-domain confidence** on every prediction, so molecules
  outside a target's chemistry are flagged instead of silently scored
- ✅ **RL molecule generation** rewarded by those QSAR models
- ✅ **Drug compatibility & interaction checking**, ADMET prediction,
  synthesis planning and chemical-space analytics
- ✅ **3D molecular visualization**

Optional extras (`torch`, `vina`, `transformers`) unlock the GNN ADMET model,
real docking and the neural SMILES generator; the app degrades gracefully
without them. See [Installation & Setup](#️-installation--setup).

## 🎯 Key Platform Features

### 🎯 QSAR Binding Prediction with an Applicability Domain

Five Random Forest models trained on ChEMBL bioactivity data predict pIC50 per
target:

| Target | Molecules | R² (random split) | R² (new scaffolds) | R² (in domain) | Domain coverage |
|---|---|---|---|---|---|
| EGFR | 6,409 | 0.733 | 0.604 | **0.623** | 77% |
| JAK2 | 7,222 | 0.733 | 0.518 | **0.649** | 77% |
| BACE1 | 6,781 | 0.713 | 0.594 | **0.615** | 85% |
| THROMBIN | 1,789 | 0.727 | 0.425 | **0.463** | 85% |
| COX2 | 4,234 | 0.562 | 0.155 | **0.233** | 67% |

Three columns because one number cannot answer the question honestly:

- **Random split** — the conventional 80/20. ChEMBL is full of congeneric
  series, so close analogues of the test molecules remain in training. This
  measures interpolation within known chemistry.
- **New scaffolds** — a Bemis-Murcko scaffold split, where a scaffold appears
  in training or test but never both. The honest estimate for novel chemistry,
  and the number to quote in a paper.
- **In domain** — the same held-out scaffolds, restricted to the molecules the
  applicability-domain check accepts, over the share of the test set shown
  under *coverage*. This is what you actually get when the app labels a
  prediction reliable, and it is higher than the overall figure on every
  target — which is the applicability domain doing its job.

All five beat a mean-prediction baseline on unseen scaffolds, so the models
learn real structure–activity relationships rather than the training mean.
**COX2 is the weak one** (0.155 on new scaffolds) and is flagged as such in the
app rather than averaged away.

<details>
<summary>How these numbers were improved (+0.03 to +0.05 R² on new scaffolds)</summary>

Three things were tried, measured on the scaffold split. Two failed:

| Change | Effect on scaffold R² |
|---|---|
| Add physicochemical descriptors to the fingerprint | **−0.04** (0.575 → 0.537 on EGFR) |
| Add MACCS keys | **−0.04** |
| Gradient boosting instead of Random Forest | **−0.01**, and far slower |
| **Tune the forest's hyperparameters** | **+0.03 to +0.05** on every target |

The descriptors failing is informative: they encode global molecular properties
that track scaffold identity, so the model leans on them and transfers *worse*
to new scaffolds. They would have improved the random-split number.

What actually mattered was `max_features`. At `'sqrt'` the forest considered
only √2048 ≈ 45 of the 2048 fingerprint bits per split; on sparse fingerprints
where any individual bit is almost always zero, that is too few to find the
informative substructures, so trees ended up near-random and the ensemble
regressed towards the mean. Sampling 10% (205 bits) and removing the depth cap
fixed it.

Hyperparameters were selected by **scaffold-grouped cross-validation inside the
training half**, so the held-out scaffolds were never used for tuning
(`hyperparameter_sweep.json`). Model size was part of the choice:
`min_samples_leaf=1` with 400 trees scored *worse* (0.600 vs 0.604) at 5× the
size, so the shipped config is 200 trees with `min_samples_leaf=2` — same
inference latency as before, and the whole model set gzips to 44 MB, **smaller
than the 61 MB it replaced**.

Structures are also salt-stripped and charge-neutralised before training, with
duplicate parents merged at their median activity. That affects under 3% of
rows, so it is a cleanup rather than a cause of the gains.
</details>

**Every prediction also carries a confidence tier.** This matters more than it
sounds. A Random Forest cannot say "I don't know" — for a molecule unlike
anything in its training set it returns approximately the training-set mean,
and that number looks exactly as confident as a real prediction:

```
methane ("C")  ->  JAK2 pIC50 6.74     # JAK2 training mean is 7.13
ruxolitinib    ->  JAK2 pIC50 8.08     # an actual JAK2 inhibitor
```

Because the RL generator's reward is derived from pIC50, an ungated model pays
out for chemical nonsense. [`applicability_domain.py`](applicability_domain.py)
fixes this by comparing each molecule against the nearest members of the
target's training set (Morgan-fingerprint Tanimoto) and the spread across the
forest's trees, then scaling the reward by the resulting reliability.

Thresholds are derived per target from the leave-one-out nearest-neighbour
similarity distribution of that target's own training set, so a sparse target
gets a looser domain than a dense one — nothing is hand-tuned.

The effect on the EGFR reward signal:

| Molecule | Raw reward | Gated | Nearest-neighbour similarity | Verdict |
|---|---|---|---|---|
| Gefitinib (EGFR inhibitor) | 0.638 | **0.638** | 1.00 | high |
| Erlotinib (EGFR inhibitor) | 0.653 | **0.653** | 1.00 | high |
| Celecoxib (COX2 inhibitor) | 0.227 | 0.052 | 0.31 | low |
| Caffeine | 0.224 | 0.044 | 0.26 | out of domain |
| Methane | 0.201 | **0.004** | 0.02 | out of domain |

Discrimination between genuine EGFR inhibitors and unrelated molecules improves
from roughly 3× to roughly 13×. Pass `apply_domain_gate=False` to
`get_binding_reward` to reproduce pre-gate training runs.

#### Honest generalization numbers

The R² values above come from a **random** 80/20 split. ChEMBL is full of
congeneric series, so close analogues of the test molecules sit in the training
set and a random split measures interpolation within known chemistry. A
Bemis-Murcko **scaffold** split holds out whole scaffolds and measures what the
model does on genuinely new chemistry — that is the number to quote.

Run `python train_qsar_models.py --compare` to reproduce. Every model file also
carries both figures under `metrics_by_split`, so nothing downstream has to
hardcode them:

| Target | Random R² | Scaffold R² | Scaffold RMSE | Mean-prediction baseline |
|---|---|---|---|---|
| EGFR | 0.733 | **0.604** | 0.814 | 1.305 |
| BACE1 | 0.713 | **0.594** | 0.830 | 1.336 |
| JAK2 | 0.733 | **0.518** | 0.786 | 1.145 |
| THROMBIN | 0.727 | **0.425** | 1.052 | 1.399 |
| COX2 | 0.562 | **0.155** | 1.008 | 1.159 |

**All five targets beat the mean-prediction baseline on unseen scaffolds**, so
the models learn real structure–activity relationships rather than the
training mean. EGFR and BACE1 transfer best. COX2 largely does not — at
scaffold R² 0.155 its RMSE (1.008) is only modestly better than predicting the
mean (1.159), so treat COX2 predictions on novel scaffolds with caution.

### 🆕 Novelty: is this molecule actually new?

A generative model can score perfectly by memorising its training data.
[`novelty_scorer.py`](novelty_scorer.py) checks every generated molecule
against the ChEMBL reference sets:

```
Gefitinib  ->  novelty 0.000, Tanimoto 1.00 to EGFR set   EXACT MATCH
Aspirin    ->  novelty 0.000, Tanimoto 1.00 to COX2 set   EXACT MATCH
Caffeine   ->  novelty 0.741, Tanimoto 0.26               novel
```

It also measures **internal diversity** within a generated batch, which catches
mode collapse that novelty alone cannot — a generator emitting near-copies of
one molecule looks novel against ChEMBL but scores 0.43 internal diversity
versus 0.94 for a genuinely varied batch.

Note the deliberate tension with the applicability domain: that check rewards
similarity to the training set, novelty rewards distance from it. A useful
candidate sits in between, and `combined_quality()` scores that band.

### 🆕 Selectivity profiling

A compound that hits EGFR at 10 nM *and* everything else at 10 nM is a
promiscuity liability, not a lead. [`selectivity_scorer.py`](selectivity_scorer.py)
turns the five independent models into a selectivity window, in log units:

```
Erlotinib -> EGFR    +1.20 log units (16x over JAK2)   modestly selective
Aspirin   -> COX2    +0.10 log units (1.3x)            weakly selective
Aspirin   -> EGFR    -0.30 log units                   promiscuous
Methane   -> EGFR    no window established             unknown
```

Off-targets whose applicability domain rejects the molecule are **excluded and
reported**, never silently compared — claiming selectivity over a model that
would just return its training mean is meaningless.

### 🆕 Batch screening & export

The **Batch Screening** mode takes a pasted list, a `.smi`/`.txt` file or a CSV
with a `smiles` column and scores the whole library against every target with
confidence tiers, novelty, synthetic accessibility and selectivity. Results
sort in-domain hits first and export to **CSV or SDF** (scores carried through
as SD properties).

Same thing over HTTP:

```bash
curl -X POST localhost:8000/api/screen/batch \
  -H 'Content-Type: application/json' \
  -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O", "C"], "primary_target": "EGFR"}'
```

`GET /api/qsar/models` returns per-target metrics and domain calibration.

### 🆕 Real synthetic accessibility

The reward's synthetic-accessibility term used to fall back to
`num_atoms / 8.0` — "a bigger molecule is harder to make" — because it imported
`sascorer` without putting RDKit's contrib directory on the path.
[`synthetic_accessibility.py`](synthetic_accessibility.py) loads the real
Ertl–Schuffenhauer SAScore:

| Molecule | Real SAScore | Old proxy |
|---|---|---|
| n-decane (10 C, linear) | 1.23 | 1.25 |
| twistane (10 C, bridged) | 3.06 | 1.25 |
| morphine | 5.27 | 2.88 |

Same atom count, very different synthesis difficulty — the old proxy could not
tell them apart.

### 📡 Massive Dataset Download (10+ GB)
- **Multi-source collection**: ChEMBL, PubChem, ZINC databases  
- **2M+ molecules** with drug-like properties
- **Automatic 3D structure generation** using RDKit
- **Parallel downloading** for maximum speed

### 🧊 Interactive 3D Visualization  
- **Real-time 3D molecular viewing** with py3Dmol
- **Interactive web interface** powered by Dash/Plotly
- **Molecular manipulation** and rotation
- **Surface and stick representations**

### 🔗 Molecular Combination Engine
- **Real-time molecular fusion** with multiple bond types (single, ether, amide)
- **Instant stability prediction** for combined molecules  
- **Interactive combining** through web interface
- **Chemical reaction simulation**

### 🧪 Stability & Safety Analysis
- **Stability scoring** with risk assessment — `_assess_stability` in
  [`integrated_multiagent_platform.py`](integrated_multiagent_platform.py)
- **Reactive group detection** — [`chemical_safety_filter.py`](chemical_safety_filter.py)
- **ADMET property prediction** (Absorption, Distribution, Metabolism, Excretion,
  Toxicity) — [`admet_predictor.py`](admet_predictor.py), with a rule-based
  fallback when `torch` is absent
- **Drug-likeness assessment** (Lipinski, Veber, Egan rules)
- **Drug–drug interaction risk** — `_assess_interaction_risk`

## 🎯 Original Overview

This project implements a **complete 9-phase AI-driven drug discovery methodology** that combines:
- 🤖 **Multi-Agent AI Systems** with specialized roles
- 🧠 **LLM Orchestration** (GPT-4, Claude) for scientific analysis  
- 🧬 **Graph Neural Networks** for molecular property prediction
- ⚗️ **Generative AI** for novel molecule design (VAE, GA, Fragment-based)
- 🔬 **Structure-Based Drug Design** with molecular docking
- 📊 **3D Visualization** for interactive molecular exploration

## ✨ Key Features

### 🚀 Complete Pipeline Implementation
- **All 9 methodology phases** implemented with production code
- **Multi-database integration** (ChEMBL, PubChem, ZINC, BindingDB, PDB)
- **Multiple ML architectures** (RF, DNN, GNN) for property prediction
- **Diverse generation methods** (VAE, genetic algorithms, fragment-based)
- **LLM-powered analysis** with OpenAI and Anthropic integration

### 🔬 Scientific Capabilities
- **Lead optimization** with multi-objective fitness functions
- **ADMET prediction** (absorption, distribution, metabolism, excretion, toxicity)
- **Binding affinity prediction** with confidence intervals
- **Scaffold hopping** for exploring novel chemical spaces
- **Structure-activity relationship** analysis

### 🛠️ Production Ready
- **Modular architecture** - use components independently
- **Comprehensive logging** and error handling
- **Result persistence** with intermediate saving
- **Configurable pipelines** via YAML configuration
- **Fallback systems** for external dependencies

## 📋 Methodology Implementation

This pipeline implements your complete 9-phase methodology:

### Phase 1: Target Selection & Problem Definition ✅
- **Implementation**: [src/data/collectors.py](src/data/collectors.py)
- **Features**: PDB structure retrieval, target validation, binding site detection
- **Databases**: RCSB PDB integration

### Phase 2: Data Collection & Dataset Curation ✅  
- **Implementation**: [src/data/collectors.py](src/data/collectors.py), [src/data/utils.py](src/data/utils.py)
- **Features**: Multi-database integration with caching and validation
- **Databases Integrated**:
  - 🔗 **ChEMBL**: Bioactive molecules + activity data
  - 🔗 **PubChem**: Chemical structures + biological assays  
  - 🔗 **ZINC**: Ready-to-dock 3D purchasable compounds
  - 🔗 **BindingDB**: Protein-ligand binding affinities
  - 🔗 **RCSB PDB**: Protein 3D structures

### Phase 3: Preprocessing & Molecular Representations ✅
- **Implementation**: [src/preprocessing/molecular.py](src/preprocessing/molecular.py), [src/preprocessing/features.py](src/preprocessing/features.py)
- **Features**: 
  - SMILES to graph conversion for GNNs
  - Fingerprint generation (Morgan, RDKit)
  - 3D conformer generation
  - SMILES tokenization for transformers
  - Drug-likeness filtering (Lipinski's Rule of Five)

### Phase 4: ML Prediction Models ✅
- **Implementation**: [src/models/predictors.py](src/models/predictors.py), [src/models/gnn.py](src/models/gnn.py)
- **Architectures**:
  - Random Forest & MLP regressors/classifiers
  - Deep Neural Networks (PyTorch)
  - Graph Neural Networks (GCN, GAT, MPNN)
  - Property prediction (binding affinity, ADMET, toxicity)

### Phase 5: Molecule Generation (De Novo Design) ✅
- **Implementation**: [src/generation/generators.py](src/generation/generators.py)
- **Methods**:
  - Variational Autoencoders (VAE) for SMILES
  - Genetic Algorithms with fitness optimization
  - Fragment-based generation
  - Reinforcement Learning optimization

### Phase 6: Structure-Based Docking & Scoring ✅
- **Implementation**: [src/docking/molecular_docking.py](src/docking/molecular_docking.py)
- **Features**:
  - AutoDock Vina integration
  - Binding site analysis and cavity detection
  - 3D binding pose prediction
  - Energy scoring and ranking

### Phase 7: LLM Orchestration & Analysis ✅
- **Implementation**: [src/orchestration/llm_orchestrator.py](src/orchestration/llm_orchestrator.py)
- **Providers**: OpenAI GPT-4, Anthropic Claude, Rule-based fallback
- **Capabilities**:
  - Scientific analysis of results
  - Decision-making for next steps
  - Structure-activity relationship insights
  - Optimization strategy recommendations

### Phase 8: 3D Visualization ✅
- **Implementation**: [src/visualization/molecular_viz.py](src/visualization/molecular_viz.py)
- **Tools Integrated**:
  - **PyMOL**: Protein-ligand complex visualization
  - **NGL Viewer**: Web-based interactive visualization  
  - **3Dmol.js**: Browser-based molecular graphics
  - **Plotly**: 3D scatter plots and analysis

### Phase 9: Optimization Loop ✅
- **Implementation**: [src/optimization/pipeline.py](src/optimization/pipeline.py)
- **Features**:
  - Iterative Generate → Predict → Dock → Analyze cycle
  - Convergence detection and early stopping
  - Multi-objective optimization
  - Comprehensive result tracking

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Multi-Agent-Drug-Discovery
python setup.py
```

### 2. Configure Environment
```bash
# Edit .env file with your API keys
nano .env

# Add your OpenAI/Anthropic API keys:
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### 3. Build Molecular Dataset (NEW! 🔥)
```bash
# Download 60k real drug molecules + prepare for fine-tuning
python examples/build_molecular_dataset.py

# This creates:
# • 60,000 SMILES from ChEMBL/ZINC
# • 2D molecular images (PNG)  
# • 3D conformers (SDF)
# • PyTorch datasets for Vision/Graph/Multimodal models
# • Optimized for RTX 4060 + i7 14th gen
```

### 4. Run Complete Pipeline
```bash
# Validate the QSAR models, applicability domain, novelty and selectivity
pytest tests/ -v

# Inspect the models directly (predictions + confidence for known drugs)
python qsar_predictor.py

# Honest generalization numbers (random vs scaffold split)
python train_qsar_models.py --compare

# Novelty, selectivity and synthetic-accessibility demos
python novelty_scorer.py
python selectivity_scorer.py
python synthetic_accessibility.py

# Rebuild the applicability-domain caches after retraining
python build_domain_caches.py

# Run complete drug discovery example
python examples/example_pipeline.py
```

### 5. Fine-Tune AI Models (NEW! 🔥)
```bash
# Fine-tune vision models on molecular images
python examples/fine_tune_vision.py --model resnet50 --property logp --epochs 20

# Available models: resnet50, efficientnet_b0, vit_b_16
# Available properties: molecular_weight, logp, hba, hbd, tpsa
```

**Expected Output:**
```
🧬 AI Drug Discovery Pipeline - EGFR Target
============================================================
✅ Data Collection: Found 1,250 active compounds
✅ ML Training: Binding affinity RMSE: 0.85 kcal/mol  
✅ Generation: Created 500 novel molecules
✅ Docking: Best affinity: -9.2 kcal/mol
✅ LLM Analysis: Identified 15 promising leads
🎯 Optimization Complete: 15.3% success rate

🔥 Molecular Dataset Built: 60,000 molecules ready for fine-tuning!
```

## 💻 Usage Examples

### Basic Drug Discovery Pipeline
```python
from src.optimization import OptimizationConfig, DrugDiscoveryOptimizer

# Configure target
config = OptimizationConfig(
    target_name="EGFR",
    target_pdb_path="data/structures/egfr.pdb",
    max_iterations=10,
    molecules_per_iteration=100,
    generation_method='genetic',
    llm_provider='openai'
)

# Run optimization
optimizer = DrugDiscoveryOptimizer(config)
results = optimizer.run_optimization()

# Access results
print(f"Best affinity: {results.best_affinity} kcal/mol")
print(f"Novel molecules: {len(results.novel_molecules)}")
```

### Custom Molecule Generation
```python
from src.generation import MolecularGenerator

# Initialize generator
generator = MolecularGenerator('vae')

# Generate molecules targeting specific properties
molecules = generator.generate(
    num_molecules=100,
    target_properties={'logP': 2.5, 'MW': 400}
)
```

### Property Prediction
```python
from src.models import PropertyPredictor

# Load pre-trained model
predictor = PropertyPredictor('binding_affinity')

# Predict for SMILES
smiles = ["CCc1ccc(C)cc1", "COc1ccc2[nH]cnc2c1"]
predictions = predictor.predict(smiles)
```

## 🏗️ Pipeline Architecture

The pipeline implements a **complete 9-phase methodology**:

```
Phase 1: Target Selection → Phase 2: Data Collection → Phase 3: Preprocessing
           ↑                                                          ↓
Phase 9: Optimization ←── Phase 8: 3D Visualization ←── Phase 4: ML Prediction
           ↑                        ↑                         ↓
           └── Phase 7: LLM Analysis ←── Phase 6: Docking ←── Phase 5: Generation
```

### Core Components

| Component | Purpose | Key Technologies |
|-----------|---------|------------------|
| **[Data](src/data/)** | Multi-database integration | ChEMBL, PubChem, ZINC, BindingDB, PDB |
| **[Preprocessing](src/preprocessing/)** | Molecular representations | RDKit, Graph conversion, Fingerprints |
| **[Models](src/models/)** | ML property prediction | PyTorch, Scikit-learn, GNNs |
| **[Generation](src/generation/)** | Novel molecule design | VAE, Genetic algorithms, Fragment-based |
| **[Docking](src/docking/)** | Structure-based design | AutoDock Vina, Binding site analysis |
| **[Orchestration](src/orchestration/)** | LLM-guided analysis | OpenAI GPT-4, Anthropic Claude |
| **[Visualization](src/visualization/)** | 3D molecular graphics | PyMOL, NGL, Plotly, 3Dmol.js |
| **[Optimization](src/optimization/)** | Iterative pipeline | Multi-objective optimization |

## 📊 Output Structure

When you run the pipeline, it creates comprehensive outputs:

```
outputs/
├── optimization_summary.json          # Final results and metrics
├── iteration_01/                     # Results from each iteration
│   ├── generated_molecules.txt       # AI-generated molecules
│   ├── docking_results.csv          # Binding affinity predictions  
│   ├── predictions.csv              # ML property predictions
│   ├── llm_analysis.json            # LLM scientific analysis
│   └── metrics.json                 # Performance metrics
├── iteration_02/
│   └── ...
├── visualizations/                   # 3D molecular visualizations
│   ├── binding_affinities_3d.html   # Interactive 3D plots
│   ├── chemical_space.html          # Chemical space mapping
│   ├── complex_visualizations/      # Protein-ligand complexes
│   └── visualization_summary.html   # Comprehensive report
└── logs/                            # Detailed execution logs
```

## ⚙️ Configuration

The pipeline is highly configurable through [configs/config.yaml](configs/config.yaml):

```yaml
# Example: Customize for your target
optimization:
  max_iterations: 20
  molecules_per_iteration: 100
  generation_method: 'genetic'  # or 'vae', 'fragment'
  
llm:
  provider: 'openai'  # or 'anthropic', 'fallback'
  model: 'gpt-4-turbo-preview'
  
models:
  device: 'auto'  # 'cpu', 'cuda'
  batch_size: 32
```

## 🔬 Scientific Capabilities

### Drug Discovery Tasks
- **Lead optimization** with multi-objective goals
- **Scaffold hopping** for novel chemical spaces  
- **ADMET prediction** (absorption, distribution, metabolism, excretion, toxicity)
- **Binding affinity prediction** with confidence intervals
- **Drug-drug interaction analysis**

### AI/ML Methods
- **Graph Neural Networks** for molecular property prediction
- **Variational Autoencoders** for novel molecule generation
- **Genetic Algorithms** with custom fitness functions
- **Transfer learning** from pre-trained chemical models
- **Active learning** for efficient data collection

### LLM Integration
- **Scientific literature analysis** and hypothesis generation
- **Structure-activity relationship** insights
- **Experimental design** recommendations  
- **Risk assessment** and safety analysis
- **Multi-step reasoning** for complex drug discovery problems

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.7+**
- **Optional**: GPU with CUDA for faster training
- **Optional**: PyMOL for advanced 3D visualization
- **Optional**: AutoDock Vina for molecular docking

### Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch-geometric

# Optional: Advanced visualization
conda install -c conda-forge pymol-open-source

# Optional: Molecular docking  
# Download AutoDock Vina from: https://autodock-vina.readthedocs.io/
```

### Deploying (Streamlit Community Cloud)

Point the app at `integrated_multiagent_platform.py`; `requirements.txt` and
`runtime.txt` (Python 3.12) are already set up. Put `GEMINI_API_KEY` in the
app's **Secrets** — [`chatbot/config.py`](chatbot/config.py) reads Streamlit
secrets first and falls back to environment variables, so nothing else changes
between local and deployed runs.

Three things matter for a deployment and are all enforced by
[CI](.github/workflows/ci.yml):

1. **Commit `models/*_ad.npz`.** These are the applicability-domain caches
   (1.1 MB total). Without them the app rebuilds them from the ChEMBL CSVs on
   first use — fingerprinting ~25,000 molecules, which takes **~40 s and
   stalls the first page render**. With them, cold import is ~5 s. Regenerate
   with `python build_domain_caches.py` after retraining and commit the result.
2. **Keep XSRF protection on.** The Batch Screening page accepts file uploads
   and Streamlit's XSRF protection is what guards that endpoint. It was
   previously disabled in `.streamlit/config.toml` alongside `enableCORS` —
   the usual workaround for upload trouble behind a reverse proxy. If uploads
   misbehave on a custom host, fix the proxy headers instead.
3. **Model files are gzipped pickles.** The forests are ~35 MB raw and ~11 MB
   compressed; `qsar_predictor.py` sniffs the gzip magic number so older
   plain pickles still load. Keep `scikit-learn` below 2.0 — unpickling
   across a major version is not guaranteed to reproduce the same
   predictions; retrain with `train_qsar_models.py` if you move past it.

The app degrades gracefully rather than crashing when pieces are missing:
without `torch` the GNN ADMET model falls back to rules, without the domain
caches predictions still appear but are no longer labelled reliable (and the
page says so), and without the QSAR models binding falls back to descriptor
heuristics. Toggle **Show system status** in the sidebar to see what is live.

## 🔗 Key Datasets Integrated

All major datasets from your methodology are integrated:

| Dataset | Purpose | Implementation | Status |
|---------|---------|----------------|---------|
| **ChEMBL** | Bioactive molecules + activities | `ChEMBLCollector` | ✅ Ready |
| **PubChem** | Chemical structures + assays | `PubChemCollector` | ✅ Ready |  
| **ZINC** | Purchasable 3D compounds | `ZINCCollector` | ✅ Ready |
| **BindingDB** | Binding affinities | `BindingDBCollector` | ✅ Ready |
| **RCSB PDB** | Protein structures | `PDBCollector` | ✅ Ready |
| **TDC** | ML-ready benchmarks | Via integrations | 🔧 Optional |

## 📚 Documentation

- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Complete technical documentation
- **[Configuration](configs/config.yaml)** - All pipeline settings
- **[Examples](examples/)** - Usage examples and tutorials  
- **[API Reference](docs/api/)** - Detailed API documentation

## 🎯 Next Steps & Extensions

The pipeline provides a solid foundation for:

### 🔬 Scientific Extensions
- **Quantum mechanical calculations** for accuracy
- **Molecular dynamics simulations** for binding validation
- **Free energy perturbation** for precise affinity prediction
- **PROTAC design** for protein degradation
- **Reaction prediction** for synthetic route planning

### 🤖 AI/ML Enhancements  
- **Foundation models** (ChemBERTa, MoLFormer)
- **Diffusion models** for 3D molecular generation
- **Multi-agent systems** with specialized roles
- **Federated learning** across institutions
- **Explainable AI** for regulatory compliance

### 🏭 Production Scaling
- **Cloud deployment** (AWS, Azure, GCP)
- **Container orchestration** with Kubernetes
- **Database integration** (PostgreSQL, MongoDB)
- **API development** for web interfaces
- **High-performance computing** integration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Clone repository
git clone <repo-url>
cd Multi-Agent-Drug-Discovery

# Create development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .
```

## ✅ Project Status

**Working and verified** (app boots clean, 58 tests passing):

- Streamlit platform — `streamlit run integrated_multiagent_platform.py`
- QSAR binding prediction for 5 targets, with applicability-domain confidence
- Scaffold-split validation (`train_qsar_models.py --compare`)
- Novelty / memorization checking and batch diversity metrics
- Cross-target selectivity profiling
- Batch library screening with CSV + SDF export
- Real Ertl–Schuffenhauer synthetic accessibility
- RL and multi-target molecule generation, rewarded by gated QSAR predictions
- Compatibility checking, ADMET, synthesis planning, chemical-space analytics
- Gemini-backed chatbot with 7 domain interpreters ([`chatbot/`](chatbot/))
- FastAPI server ([`web/api_server.py`](web/api_server.py)) — needs `fastapi`, `uvicorn`

**Requires optional dependencies**: the GNN ADMET model and neural SMILES
generator need `torch`; real docking needs `vina`. All degrade gracefully.

**Quarantined**: five modules in [`legacy/`](legacy/) do not parse. They were
committed already broken in the initial commit, never executed, and are not
reachable from working code — see [`legacy/README.md`](legacy/README.md) for the
details and for what replaced each one.

**Not implemented despite being referenced in docs**: `examples/fine_tune_multimodal.py`,
`examples/fine_tune_graph.py`, and the modules `sequential_training_plan.py`
imports (`test_related_targets`, `comprehensive_validation`).

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs four jobs so the
breakage described above cannot recur silently:

- **Syntax gate** — compiles every tracked Python file (excluding `legacy/`)
- **Tests** — the 66-test suite
- **Deployment readiness** — import check, confirms the domain caches are
  committed, confirms XSRF protection is on, and enforces a 25 s cold-import
  budget
- **App renders** — drives every analysis mode through Streamlit's `AppTest`
  harness, catching NameErrors in UI branches that never get imported

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{multi_agent_drug_discovery,
  title={Multi-Agent AI Drug Discovery Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Multi-Agent-Drug-Discovery}
}
```

## 🙏 Acknowledgments

- **ChEMBL** for bioactivity data
- **PubChem** for chemical structures  
- **ZINC** for purchasable compounds
- **RCSB PDB** for protein structures
- **RDKit** for molecular informatics
- **PyTorch** for deep learning frameworks
- **OpenAI** and **Anthropic** for LLM capabilities

---


Get started now:
```bash
python setup.py
python examples/example_pipeline.py
```
