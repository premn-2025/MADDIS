"""
Section-Specific AI Chat Assistant
Uses Gemini 3 Flash Preview for intelligent responses,
with knowledge base as instant fallback.
"""
import os
import time
import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass

# Load env
try:
    from dotenv import load_dotenv
    from pathlib import Path
    _env = Path(__file__).parent / '.env'
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# Gemini availability
_gemini_model = None
_gemini_last_call = 0.0
_GEMINI_MIN_INTERVAL = 4.0  # free-tier safe spacing

def _get_gemini_model():
    """Lazy-init Gemini 3 Flash Preview (singleton)."""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    try:
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY', '')
        if not api_key or not api_key.startswith('AIza'):
            return None
        genai.configure(api_key=api_key)
        model_name = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
        _gemini_model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 512,
                "top_p": 0.8,
            }
        )
        return _gemini_model
    except Exception:
        return None

GEMINI_AVAILABLE = True
TRANSFORMERS_AVAILABLE = True

@dataclass
class SectionContext:
    section_id: str
    section_name: str
    system_prompt: str
    welcome_message: str
    example_questions: List[str]

SECTION_CONTEXTS = {
    "basic_analysis": SectionContext("basic_analysis", "Basic Molecular Analysis",
        "Pharmaceutical chemistry expert", "AI Ready!", 
        ["What is LogP?", "Lipinski rules?", "Drug-likeness?"]),
    "multiagent_analysis": SectionContext("multiagent_analysis", "Multi-Agent Drug Discovery",
        "AI drug discovery expert", "AI Ready!", 
        ["Agent coordination?", "Docking analysis?", "Why multi-agent?"]),
    "rl_generation": SectionContext("rl_generation", "RL Molecule Generation",
        "Reinforcement learning expert", "AI Ready!", 
        ["How does RL work?", "Reward functions?", "Why use RL?"]),
    "multitarget_rl": SectionContext("multitarget_rl", "Multi-Target RL",
        "Optimization expert", "AI Ready!", 
        ["Pareto optimization?", "Multiple proteins?", "Balance targets?"]),
    "chemical_space": SectionContext("chemical_space", "Chemical Space Analytics",
        "Cheminformatics expert", "AI Ready!", 
        ["Chemical space?", "Clustering methods?", "Tanimoto similarity?"]),
    "stability_analysis": SectionContext("stability_analysis", "Stability and ADMET",
        "ADMET expert", "AI Ready!", 
        ["What is ADMET?", "Metabolic stability?", "Toxicity prediction?"]),
    "drug_compatibility": SectionContext("drug_compatibility", "Drug Compatibility",
        "Drug interaction expert", "AI Ready!", 
        ["Drug interactions?", "CYP450 metabolism?", "Polypharmacy?"])
}

# Knowledge base for instant responses
KNOWLEDGE_BASE = {
    "logp": "LogP (partition coefficient) measures lipophilicity. Ideal drug LogP is 1-3. Higher values = more fat-soluble, affecting membrane permeability.",
    "lipinski": "Lipinski's Rule of 5: MW ≤500, LogP ≤5, H-donors ≤5, H-acceptors ≤10. Predicts oral bioavailability.",
    "drug-like": "Drug-likeness measures similarity to known drugs based on size, lipophilicity, and structural features.",
    "rl": "Reinforcement Learning trains AI to generate molecules by rewarding good properties (binding, safety). The agent learns to optimize structures.",
    "reward": "RL rewards combine binding affinity, drug-likeness (QED), synthetic accessibility, and ADMET scores.",
    "agent": "Multi-agent systems use specialized AI (Designer, Docker, Predictor) that collaborate for comprehensive evaluation.",
    "docking": "Molecular docking predicts ligand-protein binding. It calculates poses and estimates binding energy (kcal/mol).",
    "admet": "ADMET = Absorption, Distribution, Metabolism, Excretion, Toxicity. Determines if drug reaches target safely.",
    "toxicity": "Toxicity prediction identifies harmful molecules via structural alerts and ML models.",
    "pareto": "Pareto optimization balances multiple objectives. Solutions on the Pareto front are optimal trade-offs.",
    "chemical space": "Chemical space = all possible molecules (~10^60 drug-like). Drug discovery explores this vast space.",
    "clustering": "Molecular clustering groups similar compounds using fingerprints (k-means, t-SNE).",
    "tanimoto": "Tanimoto coefficient measures similarity (0-1). Values >0.7 suggest similar biological activity.",
    "cyp450": "CYP450 enzymes metabolize 75% of drugs. Key forms: CYP3A4, CYP2D6, CYP2C9.",
    "interaction": "Drug interactions occur via enzyme competition, protein binding, or transporter interference."
}

def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini 3 Flash Preview with free-tier throttling."""
    global _gemini_last_call
    model = _get_gemini_model()
    if model is None:
        return None
    try:
        # Free-tier throttle
        elapsed = time.time() - _gemini_last_call
        if elapsed < _GEMINI_MIN_INTERVAL:
            time.sleep(_GEMINI_MIN_INTERVAL - elapsed)
        _gemini_last_call = time.time()
        resp = model.generate_content(prompt)
        if resp and resp.text:
            return resp.text.strip()
    except Exception:
        pass  # fall through to knowledge base
    return None


def get_ai_response(user_message: str, section: SectionContext, file_content: str = None) -> str:
    """Generate response using Gemini 3, with knowledge-base fallback."""
    msg_lower = user_message.lower()

    # Build a compact Gemini prompt with section context
    system_block = (
        f"You are a {section.system_prompt} for the MADDIS drug discovery platform. "
        f"Section: {section.section_name}. "
        "Answer concisely in 2-5 sentences. Be scientific and factual. "
        "If the user asks about analysis results that don't exist yet, tell them to run the analysis first."
    )
    if file_content:
        system_block += f"\n\nUploaded data (first 800 chars):\n{file_content[:800]}"

    prompt = f"{system_block}\n\nUser question: {user_message}\n\nAnswer:"

    gemini_answer = _call_gemini(prompt)
    if gemini_answer:
        return gemini_answer

    # ---------- Fallback: local knowledge base ----------
    for keyword, response in KNOWLEDGE_BASE.items():
        if keyword in msg_lower:
            return response

    # Section-specific fallbacks
    fallbacks = {
        "basic_analysis": "Key drug properties include LogP (lipophilicity), molecular weight, and hydrogen bonding capacity. These affect absorption and activity.",
        "multiagent_analysis": "Multi-agent systems use specialized AI agents for molecular design, docking, property prediction, and validation - working together for comprehensive analysis.",
        "rl_generation": "Reinforcement learning optimizes molecules by rewarding desired properties like binding affinity and drug-likeness. The agent learns to improve structures iteratively.",
        "multitarget_rl": "Multi-target optimization finds molecules that balance activity across multiple proteins using Pareto fronts - crucial for complex diseases.",
        "chemical_space": "Chemical space analysis maps molecular diversity using fingerprints and clustering to identify unexplored regions.",
        "stability_analysis": "ADMET predicts drug behavior: absorption, distribution, metabolism, excretion, toxicity - key for clinical success.",
        "drug_compatibility": "Drug compatibility considers CYP450 metabolism, protein binding, and receptor interactions to avoid harmful combinations."
    }

    return fallbacks.get(section.section_id, "Please ask about specific topics like LogP, ADMET, docking, or molecular optimization.")


def render_chat_expander(section_id: str, molecule_context: Dict = None, key_suffix: str = ""):
    """Render chat UI with proper message display"""
    section = SECTION_CONTEXTS.get(section_id)
    if not section:
        st.error(f"Unknown section: {section_id}")
        return
    
    # Unique keys for this section
    base_key = f"{section_id}{key_suffix}"
    messages_key = f"msgs_{base_key}"
    
    # Initialize message history
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    
    # Check if we have messages to show
    has_history = len(st.session_state[messages_key]) > 0
    
    with st.expander(f"💬 Ask AI about {section.section_name}", expanded=True):
        # Status
        if _get_gemini_model() is not None:
            st.success("✅ Gemini 3 Flash Preview — AI Assistant Ready")
        else:
            st.info("ℹ️ AI Assistant Ready (offline knowledge base)")
        
        # Display chat history FIRST (before buttons/forms)
        if st.session_state[messages_key]:
            st.markdown("### 💬 Chat History")
            for i, msg in enumerate(st.session_state[messages_key]):
                if msg["role"] == "user":
                    st.info(f"👤 **You:** {msg['content']}")
                else:
                    st.success(f"🤖 **AI:** {msg['content']}")
            st.markdown("---")
        
        # File upload (optional)
        uploaded = st.file_uploader("📁 Upload file (optional)", type=["txt", "csv", "smi"], key=f"file_{base_key}")
        file_text = None
        if uploaded:
            try:
                file_text = uploaded.read().decode('utf-8')
                st.caption(f"✓ Loaded: {uploaded.name}")
            except:
                pass
        
        # Quick question buttons
        st.write("**💡 Quick Questions:**")
        button_cols = st.columns(len(section.example_questions))
        
        for idx, (col, question) in enumerate(zip(button_cols, section.example_questions)):
            btn_key = f"qbtn_{base_key}_{idx}"
            if col.button(question, key=btn_key):
                # Generate response immediately
                answer = get_ai_response(question, section, file_text)
                # Store in session
                st.session_state[messages_key].append({"role": "user", "content": question})
                st.session_state[messages_key].append({"role": "assistant", "content": answer})
                # Force refresh to show new messages
                st.rerun()
        
        # Custom input - using container to show input before rerun
        user_q = st.text_input("✍️ Your question:", key=f"input_{base_key}", placeholder="Ask anything about drug discovery...")
        
        col1, col2 = st.columns([1, 4])
        if col1.button("🚀 Send", key=f"send_{base_key}", type="primary"):
            if user_q and user_q.strip():
                answer = get_ai_response(user_q, section, file_text)
                st.session_state[messages_key].append({"role": "user", "content": user_q})
                st.session_state[messages_key].append({"role": "assistant", "content": answer})
                st.rerun()
            else:
                st.warning("Please enter a question first!")
        
        # Clear button
        if col2.button("🗑️ Clear Chat", key=f"clear_{base_key}"):
            st.session_state[messages_key] = []
            st.rerun()
