"""
Optimized Prompt Templates for Drug Discovery Chatbot

Grounded, concise prompts designed for accuracy and minimal hallucination.
"""

from typing import Any

# ==================== Core System Prompt ====================

SYSTEM_PROMPT_BASE = """You are a scientific assistant for drug discovery. 
RULES:
- Use ONLY data from the provided context
- State exact values (numbers, SMILES, scores)
- If data unavailable, say so clearly
- Be concise: 2-4 sentences maximum
- Never estimate or infer missing values"""

# ==================== Workflow-Specific Prompts ====================

WORKFLOW_PROMPTS = {
    'rl_generation': """Focus: RL molecule generation results.
Interpret: rewards, molecule scores, training progress.
Key metrics: total_reward, binding_affinity, drug_likeness, validity.""",

    'multi_target_rl': """Focus: Multi-target optimization.
Interpret: Pareto solutions, target affinities, trade-offs.
Key metrics: target_affinities, qed_score, pareto_rank.""",

    'multi_agent': """Focus: Multi-agent analysis.
Interpret: agent contributions, consensus, recommendations.
Key metrics: agent scores, status, outputs.""",

    'property_prediction': """Focus: ADMET/drug-likeness predictions.
Interpret: toxicity, solubility, permeability, Lipinski.
Key metrics: qed_score, lipinski_violations, toxicity_score.""",

    'docking': """Focus: Molecular docking results.
Interpret: binding affinity, interactions, confidence.
Key metrics: kcal/mol, H-bonds, hydrophobic contacts.""",

    'chemical_space': """Focus: Chemical diversity analysis.
Interpret: clusters, similarity, coverage.
Key metrics: tanimoto, diversity_score, num_clusters."""
}


def get_system_prompt(workflow_type: str = None) -> str:
    """Get optimized system prompt for workflow"""
    base = SYSTEM_PROMPT_BASE
    
    if workflow_type and workflow_type in WORKFLOW_PROMPTS:
        return f"{base}\n{WORKFLOW_PROMPTS[workflow_type]}"
    
    return base


# ==================== Response Formatting ====================

def format_metric_response(metric_name: str, value: Any, unit: str = "") -> str:
    """Format a metric for clear display"""
    if isinstance(value, float):
        formatted = f"{value:.2f}"
    else:
        formatted = str(value)
    
    if unit:
        return f"{metric_name}: {formatted} {unit}"
    return f"{metric_name}: {formatted}"
