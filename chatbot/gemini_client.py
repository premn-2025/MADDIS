"""
Optimized Gemini API Client for Drug Discovery Chatbot

Production-grade wrapper with:
- Low temperature for factual, grounded responses
- Optimized token limits for speed
- Strict grounding prompts to prevent hallucination
- Efficient context compression
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import get_config, ChatbotConfig

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)


class GeminiClient:
    """
    Optimized Gemini API client with grounding and fast response.
    Now includes Ollama fallback for hybrid mode.
    """
    
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """Initialize the Gemini client"""
        self.config = config or get_config()
        self.model = None
        self.is_available = False
        self.conversation_history: List[Message] = []
        self._initialization_attempted = False
        
        # Optimized settings for accuracy
        self.temperature = 0.1  # Very low for factual responses
        self.max_output_tokens = 1024  # Detailed answers
        self.top_p = 0.8  # Focused token selection
        self.top_k = 20  # Limited vocabulary for precision
        
        # Free-tier rate limiting (Gemini free tier: ~15 RPM)
        self._last_call_time = 0.0
        self._min_call_interval = 4.0  # seconds between API calls (safe for free tier)
        
        # Content-based request deduplication
        # Cache the last response; return it for identical repeat messages
        self._last_request_message = ""
        self._last_response = ""
        
        # Ollama fallback (lazy loaded)
        self._ollama_client = None
        
        # Groq client (lazy loaded)
        self._groq_client = None
        
        # Don't initialize immediately - wait for first use (lazy init)
        # This prevents rate limit errors on page load
    
    @property
    def groq_client(self):
        """Lazy-load Groq client"""
        if self._groq_client is None:
            try:
                from .groq_provider import GroqClient
                self._groq_client = GroqClient()
                if self._groq_client.is_available:
                    logger.info("Groq client ready")
            except Exception as e:
                logger.debug(f"Groq not available: {e}")
        return self._groq_client
    
    @property
    def ollama_client(self):
        """Lazy-load Ollama client for fallback"""
        if self._ollama_client is None:
            try:
                from .local_model_provider import OllamaClient
                self._ollama_client = OllamaClient()
                if self._ollama_client.is_available:
                    logger.info("Ollama fallback client ready")
            except Exception as e:
                logger.debug(f"Ollama not available: {e}")
        return self._ollama_client
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini API client"""
        self._initialization_attempted = True
        try:
            import google.generativeai as genai
            
            is_valid, error_msg = self.config.validate()
            if not is_valid:
                logger.warning(f"Gemini configuration invalid: {error_msg}")
                self.is_available = False
                return
            
            genai.configure(api_key=self.config.gemini_api_key)
            
            # Use model from config (.env GEMINI_MODEL or fallback)
            try:
                model_name = self.config.gemini_model or "gemini-3-flash-preview"
                logger.info(f"Initializing Gemini with model: {model_name}")
                self.model = genai.GenerativeModel(
                    model_name,
                    generation_config={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "max_output_tokens": self.max_output_tokens
                    }
                )
                logger.info(f"Gemini client initialized with primary model: {model_name}")
            except Exception as e:
                logger.warning(f"Primary model {model_name} failed: {e}. Trying fallback 'gemini-1.5-flash'")
                self.model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    generation_config={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "max_output_tokens": self.max_output_tokens
                    }
                )

            
            self.is_available = True
            logger.info(f"Gemini client initialized: {self.config.gemini_model}")
            
        except ImportError:
            logger.error("google-generativeai package not installed")
            self.is_available = False
        except Exception as e:
            error_str = str(e).lower()
            # Better rate limit detection during initialization
            if any(keyword in error_str for keyword in ['quota', 'rate', 'limit', '429', 'resource_exhausted']):
                logger.warning(f"API rate limit during initialization: {e}")
            else:
                logger.error(f"Failed to initialize Gemini client: {e}")
            self.is_available = False
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        include_history: bool = True
    ) -> str:
        """
        Generate a grounded, accurate response
        
        Args:
            user_message: The user's input message
            system_prompt: System instructions for the model
            context: Workflow results context (REQUIRED for accuracy)
            include_history: Whether to include last 3 exchanges
            
        Returns:
            Grounded response based on provided context
        """
        # Content-based deduplication: if the exact same message, return cached response
        if (user_message == self._last_request_message and self._last_response):
            logger.debug("Returning cached response (same message)")
            return self._last_response
        
        # Check if we should use Groq as PRIMARY
        if self.config.model_provider == 'groq':
            return self._generate_with_groq_primary(user_message, system_prompt, context)
        
        # Check if we should use Ollama as PRIMARY (not fallback)
        if self.config.model_provider == 'ollama':
            return self._generate_with_ollama_primary(user_message, system_prompt, context)
        
        # Default: Use Gemini (provider=gemini or hybrid)
        # For 'gemini' and 'hybrid', use Gemini primary with Groq fallback
        return self._generate_with_gemini(user_message, system_prompt, context, include_history)
    
    def _generate_with_groq_primary(
        self, user_message: str, system_prompt: str, context: str
    ) -> str:
        """Use Groq as the primary model (ultra-fast!)"""
        if self.groq_client and self.groq_client.is_available:
            logger.info("🚀 Using Groq as primary model")
            try:
                response_text = self.groq_client.generate_response(
                    user_message=user_message,
                    system_prompt=system_prompt,
                    context=context
                )
                if response_text and not response_text.startswith("⚠️"):
                    # Cache response (content-based, not time-based)
                    self._last_request_message = user_message
                    self._last_response = response_text
                    self._update_history(user_message, response_text)
                    return response_text
            except Exception as e:
                logger.error(f"Groq primary failed: {e}")
        
        # Groq not available - use hardcoded fallback
        logger.warning("Groq not available - using hardcoded fallback")
        return self._fallback_response(user_message, context)
    
    def _generate_with_ollama_primary(
        self, user_message: str, system_prompt: str, context: str
    ) -> str:
        """Use Ollama as the primary model (no Gemini at all)"""
        if self.ollama_client and self.ollama_client.is_available:
            logger.info("🤖 Using Ollama as primary model")
            try:
                response_text = self.ollama_client.generate_response(
                    user_message=user_message,
                    system_prompt=system_prompt,
                    context=context
                )
                if response_text and not response_text.startswith("⚠️"):
                    # Cache response (content-based, not time-based)
                    self._last_request_message = user_message
                    self._last_response = response_text
                    self._update_history(user_message, response_text)
                    return response_text
            except Exception as e:
                logger.error(f"Ollama primary failed: {e}")
        
        # Ollama not available - use hardcoded fallback
        logger.warning("Ollama not available - using hardcoded fallback")
        return self._fallback_response(user_message, context)
    
    def _generate_with_gemini(
        self, user_message: str, system_prompt: str, context: str, include_history: bool
    ) -> str:
        """Use Gemini as primary (with optional Ollama fallback for hybrid)"""
        # Lazy initialization - only initialize when first used
        if not self._initialization_attempted:
            self._initialize_client()
        
        if not self.is_available:
            return self._fallback_response(user_message, context)
        
        # Build optimized prompt with grounding
        prompt = self._build_grounded_prompt(
            user_message=user_message,
            system_prompt=system_prompt,
            context=context,
            include_history=include_history
        )
        
        # Generate with retry (max 2 attempts for speed)
        response_text = self._generate_with_retry(prompt, max_retries=2)
        
        # If API failed (returned empty), try Groq fallback, then Ollama
        if not response_text or response_text.strip() == "":
            # Try Groq fallback first (fast and reliable)
            if self.groq_client and self.groq_client.is_available:
                logger.info("Gemini returned empty - trying Groq fallback")
                try:
                    response_text = self.groq_client.generate_response(
                        user_message=user_message,
                        system_prompt=system_prompt,
                        context=context
                    )
                    if response_text and not response_text.startswith("⚠️"):
                        logger.info("✅ Groq fallback successful")
                except Exception as e:
                    logger.warning(f"Groq fallback failed: {e}")
                    response_text = ""
            
            # Then try Ollama fallback for hybrid mode
            if (not response_text or response_text.strip() == "") and self.config.model_provider == 'hybrid':
                if self.ollama_client and self.ollama_client.is_available:
                    logger.info("Trying Ollama fallback")
                    try:
                        response_text = self.ollama_client.generate_response(
                            user_message=user_message,
                            system_prompt=system_prompt,
                            context=context
                        )
                        if response_text and not response_text.startswith("⚠️"):
                            logger.info("✅ Ollama fallback successful")
                    except Exception as e:
                        logger.warning(f"Ollama fallback failed: {e}")
                        response_text = ""
            
            # Final fallback to hardcoded responses
            if not response_text or response_text.strip() == "":
                logger.info("All APIs failed - using hardcoded fallback")
                response_text = self._fallback_response(user_message, context)
        
        # Cache this request/response (content-based dedup)
        self._last_request_message = user_message
        self._last_response = response_text
        
        # Update history (keep only last 6 messages for efficiency)
        self._update_history(user_message, response_text)
        
        return response_text
    
    def _build_grounded_prompt(
        self,
        user_message: str,
        system_prompt: str,
        context: str,
        include_history: bool
    ) -> str:
        """Build a prompt optimized for factual, grounded responses"""
        
        # Grounding instruction (CRITICAL for reducing hallucination)
        # Allow general knowledge for educational questions, strict grounding only for results
        has_results_context = context and len(context) > 50 and "CONTEXT: No workflow" not in context
        
        if has_results_context:
            grounding_instruction = """CRITICAL INSTRUCTIONS:
1. ONLY use information from the PROVIDED CONTEXT below
2. If specific data is not in the context, say "This information is not available in the current results"
3. Be CONCISE - answer in 2-4 sentences when possible
4. Use EXACT numbers and values from the context
5. DO NOT invent or estimate values not explicitly provided
"""
        else:
            # For general knowledge questions (no results context)
            grounding_instruction = """INSTRUCTIONS:
1. Answer general drug discovery questions using your training knowledge
2. Topics you should know: Lipinski's Rule of 5, ADMET, LogP, molecular weight, QED, binding affinity, docking, SMILES, RL for molecules, Pareto optimization
3. Be CONCISE - answer in 2-4 sentences
4. Be factual and scientific
5. If asked about specific analysis results that don't exist, say "Please run an analysis first to get specific results"
"""
        
        parts = [grounding_instruction]
        
        if system_prompt:
            parts.append(f"ROLE: {system_prompt[:300]}")
        
        if context:
            # Compress context to essential data
            compressed = self._compress_context(context)
            parts.append(f"CONTEXT DATA:\n{compressed}")
        
        if include_history and self.conversation_history:
            history = self._format_history(limit=3)
            parts.append(f"RECENT CONVERSATION:\n{history}")
        
        parts.append(f"USER QUESTION: {user_message}")
        
        if has_results_context:
            parts.append("ANSWER (grounded in context only):")
        else:
            parts.append("ANSWER:")
        
        return "\n\n".join(parts)
    
    def _compress_context(self, context: str, max_chars: int = 1500) -> str:
        """Compress context to essential information"""
        if len(context) <= max_chars:
            return context
        
        # Keep structured data, remove verbose descriptions
        lines = context.split('\n')
        essential_lines = []
        char_count = 0
        
        for line in lines:
            # Prioritize lines with actual data
            if any(c in line for c in [':', '=', '-', '•', '*']) or len(line) < 50:
                if char_count + len(line) < max_chars:
                    essential_lines.append(line)
                    char_count += len(line)
        
        return '\n'.join(essential_lines)
    
    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response with free-tier-friendly rate limiting and retry"""
        import time
        
        # Free-tier throttle: wait if we called too recently
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_call_interval:
            wait = self._min_call_interval - elapsed
            logger.debug(f"Free-tier throttle: sleeping {wait:.1f}s")
            time.sleep(wait)
        
        for attempt in range(max_retries):
            try:
                self._last_call_time = time.time()
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
                
            except Exception as e:
                logger.warning(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
                error_str = str(e).lower()
                
                # Detect rate limit
                if any(kw in error_str for kw in ['quota', 'rate', 'limit', '429', 'resource_exhausted']):
                    if attempt < max_retries - 1:
                        # Exponential back-off: 10s, 30s
                        wait = 10 * (attempt + 1)
                        logger.warning(f"Rate limit hit. Waiting {wait}s before retry...")
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning("Rate limit persists. Falling back.")
                        return ""
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
        
        return ""
    
    def _format_history(self, limit: int = 3) -> str:
        """Format recent conversation history efficiently"""
        recent = self.conversation_history[-(limit * 2):]
        lines = []
        for msg in recent:
            prefix = "Q:" if msg.role == 'user' else "A:"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{prefix} {content}")
        return "\n".join(lines)
    
    def _update_history(self, user_msg: str, assistant_msg: str) -> None:
        """Update conversation history with trimming"""
        self.conversation_history.append(Message(role='user', content=user_msg))
        self.conversation_history.append(Message(role='assistant', content=assistant_msg))
        
        # Keep only last 6 messages (3 exchanges)
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
    
    def _fallback_response(self, user_message: str, context: str = "") -> str:
        """Provide factual fallback response based on context"""
        msg_lower = user_message.lower()
        
        # Try to extract specific data from context
        if context:
            if 'binding' in msg_lower or 'affinity' in msg_lower:
                if 'kcal/mol' in context:
                    import re
                    match = re.search(r'(-?\d+\.?\d*)\s*kcal/mol', context)
                    if match:
                        return f"📊 The binding affinity is {match.group(1)} kcal/mol. Values below -8.0 indicate strong binding."
            
            if 'best' in msg_lower and 'molecule' in msg_lower:
                if 'SMILES' in context:
                    import re
                    match = re.search(r'SMILES[:\s]+([^\s\n]+)', context)
                    if match:
                        return f"🧬 The best molecule SMILES: {match.group(1)[:50]}"
            
            if 'qed' in msg_lower or 'drug-like' in msg_lower:
                if 'QED' in context or 'qed' in context:
                    import re
                    match = re.search(r'[Qq][Ee][Dd][:\s]+(\d*\.?\d+)', context)
                    if match:
                        qed = float(match.group(1))
                        assessment = "excellent" if qed > 0.7 else "good" if qed > 0.5 else "needs improvement"
                        return f"💊 QED Score: {qed:.2f} ({assessment}). QED ranges 0-1, higher is more drug-like."
        
        # Comprehensive drug discovery knowledge base
        definitions = {
            # ADMET Properties
            ('admet',): "💊 **ADMET** = Absorption, Distribution, Metabolism, Excretion, Toxicity. These 5 properties determine if a drug can reach its target safely. Good ADMET = higher chance of clinical success.",
            ('absorption',): "💊 **Absorption** measures how well a drug enters the bloodstream. Oral drugs need good intestinal absorption. Key factors: lipophilicity, molecular weight, solubility.",
            ('distribution',): "💊 **Distribution** describes how a drug spreads through body tissues. Affected by protein binding, lipophilicity, and blood-brain barrier penetration.",
            ('metabolism',): "💊 **Metabolism** is how the body breaks down drugs, mainly in the liver via CYP450 enzymes. Fast metabolism = short drug action. CYP3A4, CYP2D6, CYP2C9 are key enzymes.",
            ('excretion', 'elimination'): "💊 **Excretion** is drug removal from the body, mainly via kidneys (urine) or liver (bile). Half-life measures how long a drug stays active.",
            ('toxicity', 'toxic'): "⚠️ **Toxicity** predicts harmful effects. Key concerns: hepatotoxicity (liver), cardiotoxicity (heart/hERG), mutagenicity (DNA damage), nephrotoxicity (kidney).",
            
            # Binding & Docking
            ('binding', 'affinity'): "📊 **Binding Affinity** measures drug-target binding strength in kcal/mol. Stronger binding = more negative values. <-8.0 = strong, -6 to -8 = moderate, >-6 = weak.",
            ('docking',): "🔗 **Molecular Docking** simulates how a drug fits into a protein's binding pocket. It predicts binding poses and estimates affinity. Tools: AutoDock Vina, Glide.",
            ('pose',): "🔗 **Binding Pose** is the 3D orientation of a drug in a protein pocket. Good poses maximize hydrogen bonds and hydrophobic contacts while minimizing steric clashes.",
            ('hydrogen bond', 'h-bond'): "🔗 **Hydrogen Bonds** are key drug-protein interactions. Donors (NH, OH) interact with acceptors (O, N). Strong H-bonds improve binding affinity by 1-3 kcal/mol each.",
            
            # Molecular Properties
            ('logp', 'lipophilicity'): "📊 **LogP** measures lipophilicity (fat vs water solubility). Ideal range: 1-3. Too high (>5) = poor solubility, accumulation. Too low (<0) = poor membrane permeation.",
            ('molecular weight', 'mw'): "📊 **Molecular Weight** affects absorption. Ideal: 150-500 Da. >500 Da = harder to absorb orally (Lipinski rule).",
            ('lipinski', 'rule of 5', 'ro5'): "📊 **Lipinski's Rule of 5**: MW≤500, LogP≤5, H-bond donors≤5, acceptors≤10. Violations reduce oral bioavailability. 90% of oral drugs pass these rules.",
            ('tpsa', 'polar surface'): "📊 **TPSA** (Topological Polar Surface Area) predicts membrane permeability. <140 Å² = good absorption. <90 Å² = can cross blood-brain barrier.",
            ('rotatable bonds',): "📊 **Rotatable Bonds** affect flexibility and oral bioavailability. Ideal: <10. More rotatable bonds = harder to absorb, less rigid binding.",
            ('solubility',): "📊 **Solubility** is crucial for drug absorption. Poor solubility = poor bioavailability. Measured as LogS. >-4 = good, -4 to -6 = moderate, <-6 = poor.",
            
            # Drug-likeness
            ('qed', 'drug-like', 'druglike'): "💊 **QED** (Quantitative Estimate of Drug-likeness) ranges 0-1. Combines MW, LogP, TPSA, rotatable bonds, H-bonds, alerts. >0.5 = drug-like, >0.7 = excellent.",
            ('drug score',): "💊 **Drug Score** combines multiple properties into one metric. Higher = better drug candidate. Considers binding, ADMET, synthesizability.",
            ('synthetic accessibility', 'sa score'): "⚗️ **Synthetic Accessibility** (SA Score) predicts how easy a molecule is to synthesize. Ranges 1-10. <4 = easy, 4-6 = moderate, >6 = difficult.",
            
            # RL & AI
            ('rl', 'reinforcement learning'): "🧬 **Reinforcement Learning** trains AI to generate molecules by rewarding good properties. The agent learns to optimize for binding affinity, drug-likeness, and safety.",
            ('reward', 'reward function'): "🧬 **Reward Function** in RL combines: binding affinity (docking score), drug-likeness (QED), novelty, validity, and ADMET predictions. Weights balance trade-offs.",
            ('policy', 'policy gradient'): "🧬 **Policy Gradient** is an RL algorithm that directly optimizes the molecule generation policy. REINFORCE and PPO are common variants.",
            ('generation', 'generate', 'generator'): "🧬 **Molecular Generation** uses AI to create new drug candidates. Methods: SMILES-based RNNs, graph neural networks, VAEs, diffusion models.",
            ('smiles',): "🧬 **SMILES** (Simplified Molecular Input Line Entry System) represents molecules as text strings. Example: aspirin = CC(=O)OC1=CC=CC=C1C(=O)O",
            
            # Multi-target
            ('pareto', 'multi-objective'): "📊 **Pareto Optimization** finds best trade-offs when optimizing multiple targets. Pareto-optimal solutions can't improve one objective without worsening another.",
            ('multi-target', 'polypharmacology'): "📊 **Multi-Target Drugs** bind multiple proteins simultaneously. Useful for complex diseases (cancer, neurodegeneration). Pareto optimization balances target affinities.",
            
            # Proteins & Targets
            ('protein', 'target'): "🎯 **Drug Targets** are proteins (enzymes, receptors, ion channels) that drugs bind to. Common: kinases (cancer), GPCRs (many diseases), proteases (infection).",
            ('kinase',): "🎯 **Kinases** are enzymes that add phosphate groups. Important cancer targets. Examples: EGFR, BRAF, CDK4/6. Kinase inhibitors are major drug class.",
            ('egfr',): "🎯 **EGFR** (Epidermal Growth Factor Receptor) is a key cancer target. Overactive in lung, breast, colorectal cancers. Drugs: gefitinib, erlotinib, osimertinib.",
            ('receptor',): "🎯 **Receptors** are proteins that receive signals. GPCRs (G-protein coupled receptors) are targets for ~35% of all drugs.",
            
            # Safety & Alerts
            ('herg', 'cardiotoxicity'): "⚠️ **hERG** is a heart ion channel. Blocking it causes fatal arrhythmias. All drugs must be tested for hERG liability. High LogP increases risk.",
            ('ames', 'mutagenicity'): "⚠️ **AMES Test** detects mutagenicity (DNA damage potential). Positive = potential carcinogen. Certain structural alerts (nitro groups, epoxides) are red flags.",
            ('pains', 'structural alerts'): "⚠️ **PAINS** (Pan-Assay Interference Compounds) are molecules that give false positives. They have reactive groups that non-specifically bind proteins.",
            ('bbb', 'blood-brain barrier'): "🧠 **BBB** (Blood-Brain Barrier) protects the brain. CNS drugs must cross it. Requirements: MW<450, TPSA<90, LogP 1-3, low H-bond donors.",
            
            # Synthesis
            ('synthesis', 'retrosynthesis'): "⚗️ **Retrosynthesis** plans how to make a molecule by working backwards from product to starting materials. AI tools predict synthetic routes.",
            ('route', 'synthetic route'): "⚗️ **Synthetic Route** is the step-by-step process to make a drug. Fewer steps = cheaper, faster. Key considerations: yield, stereochemistry, safety.",
            
            # General
            ('what is', 'explain', 'define', 'tell me'): "🤖 I'm your drug discovery AI assistant. Ask me about: ADMET properties, binding affinity, docking, molecular properties (LogP, MW, TPSA), drug-likeness (QED), RL generation, synthesis planning, or any specific results from your analysis!",
            ('help', 'how to', 'guide'): "🤖 **How to use me:** Ask questions about your drug discovery results! Examples: 'What is the binding affinity?', 'Explain the ADMET predictions', 'Is this molecule drug-like?', 'What is LogP?'",
            ('best', 'top', 'optimal'): "📊 To identify the **best molecule**, look for: binding affinity <-8 kcal/mol, QED >0.5, no Lipinski violations, SA score <6, no toxicity alerts. Balance all properties!",
            ('result', 'interpret', 'analysis'): "📊 To **interpret results**: Check binding affinity (more negative = better), QED score (>0.5 = drug-like), ADMET flags (green = safe), and synthetic accessibility (<6 = makeable).",
        }
        
        for keywords, response in definitions.items():
            if any(kw in msg_lower for kw in keywords):
                return response
        
        return "ℹ️ Please run an analysis first, then ask about the specific results. I can interpret RL generation, docking, ADMET, and multi-target optimization results."
    
    def clear_history(self) -> None:
        """Clear conversation history and response cache"""
        self.conversation_history = []
        self._last_request_message = ""
        self._last_response = ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "is_available": self.is_available,
            "model": self.config.gemini_model,
            "history_length": len(self.conversation_history),
            "temperature": self.temperature
        }
