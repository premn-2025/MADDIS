"""
LLM Orchestration for AI Drug Discovery

The LLM acts as the "brain" that:
- Analyzes results from different pipeline stages
- Makes decisions about next steps
- Suggests optimizations
- Provides scientific insights
- Orchestrates the overall workflow
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for LLM analysis results"""
    summary: str
    recommendations: List[str]
    confidence_score: float
    next_actions: List[str]
    insights: Dict[str, Any]


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider - PRIMARY for Gemini 3 Hackathon"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3-flash-preview"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = model
        
        if not self.api_key:
            logger.error("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            self.available = False
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
            self.available = True
            logger.info(f"✅ Gemini provider initialized with model: {model}")
        except ImportError:
            logger.error("google-generativeai library not installed. Run: pip install google-generativeai")
            self.available = False
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            self.available = False

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini API"""
        if not self.available:
            return "Gemini provider not available"
        
        try:
            # Configure generation parameters
            generation_config = {
                "max_output_tokens": kwargs.get('max_tokens', 2000),
                "temperature": kwargs.get('temperature', 0.7),
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response: {e}"


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (DEPRECATED - Use Gemini for hackathon)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Use Gemini instead for hackathon.")
            self.available = False
            return
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.available = True
            logger.info(f"OpenAI provider initialized with model: {model}")
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            self.available = False

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API"""
        if not self.available:
            return "OpenAI provider not available"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {e}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (DEPRECATED - Use Gemini for hackathon)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        
        if not self.api_key:
            logger.warning("Anthropic API key not found. Use Gemini instead for hackathon.")
            self.available = False
            return
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.available = True
            logger.info(f"Anthropic provider initialized with model: {model}")
        except ImportError:
            logger.error("Anthropic library not installed. Run: pip install anthropic")
            self.available = False

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API"""
        if not self.available:
            return "Anthropic provider not available"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error generating response: {e}"


class FallbackProvider(LLMProvider):
    """Fallback provider with rule-based responses"""

    def __init__(self):
        self.available = True
        logger.info("Fallback provider initialized (rule-based responses)")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate simple rule-based response"""
        prompt_lower = prompt.lower()
        
        if 'binding affinity' in prompt_lower or 'docking' in prompt_lower:
            return """Based on the docking results analysis:
- Molecules with binding affinity < -8.0 kcal/mol show strong binding potential
- Consider optimizing molecular weight (300-500 Da) and logP (2-4)
- Focus on hydrogen bonding interactions in the binding site
- Recommend further optimization of top 10% hits"""
        
        elif 'generation' in prompt_lower or 'new molecules' in prompt_lower:
            return """For molecule generation optimization:
- Generated molecules show reasonable chemical diversity
- Recommend filtering by drug-likeness (Lipinski's Rule of Five)
- Consider scaffold hopping to explore new chemical space
- Validate generated structures with quantum mechanical calculations"""
        
        elif 'admet' in prompt_lower or 'toxicity' in prompt_lower:
            return """ADMET analysis recommendations:
- Prioritize molecules with predicted good solubility (logS > -4)
- Monitor CYP interactions for potential drug-drug interactions
- Consider BBB permeability for CNS targets
- Recommend experimental validation for top candidates"""
        
        else:
            return """General drug discovery recommendations:
- Continue iterative optimization cycles
- Balance multiple objectives (potency, selectivity, ADMET)
- Consider structure-activity relationships (SAR)
- Validate computational predictions with experimental data"""


class DrugDiscoveryOrchestrator:
    """Main LLM orchestrator for drug discovery pipeline - Gemini 3 Hackathon Edition"""

    def __init__(self, provider: str = 'gemini', **kwargs):
        """Initialize orchestrator with LLM provider.
        
        Args:
            provider: 'gemini' (default), 'openai', 'anthropic', or 'fallback'
        """
        if provider == 'gemini':
            self.llm = GeminiProvider(**kwargs)
        elif provider == 'openai':
            self.llm = OpenAIProvider(**kwargs)
        elif provider == 'anthropic':
            self.llm = AnthropicProvider(**kwargs)
        elif provider == 'fallback':
            self.llm = FallbackProvider()
        else:
            logger.warning(f"Unknown provider {provider}, using Gemini")
            self.llm = GeminiProvider(**kwargs)
        
        if not self.llm.available:
            logger.warning("Primary LLM provider not available, using fallback")
            self.llm = FallbackProvider()

    def analyze_docking_results(self, docking_df: pd.DataFrame,
                               target_info: Optional[str] = None) -> AnalysisResult:
        """Analyze molecular docking results"""
        stats = {
            'total_molecules': len(docking_df),
            'best_affinity': docking_df['binding_affinity'].min(),
            'mean_affinity': docking_df['binding_affinity'].mean(),
            'top_10_percent_cutoff': docking_df['binding_affinity'].quantile(0.1)
        }
        
        prompt = f"""
Analyze the following molecular docking results for drug discovery:

Target Information: {target_info or "Not specified"}

Docking Statistics:
- Total molecules screened: {stats['total_molecules']}
- Best binding affinity: {stats['best_affinity']:.2f} kcal/mol
- Mean binding affinity: {stats['mean_affinity']:.2f} kcal/mol
- Top 10% cutoff: {stats['top_10_percent_cutoff']:.2f} kcal/mol

Top 10 molecules:
{docking_df.head(10)[['smiles', 'binding_affinity']].to_string()}

Please provide:
1. Scientific analysis of the results
2. Identification of promising lead compounds
3. Recommendations for structure optimization
4. Next steps in the drug discovery pipeline
5. Potential concerns or limitations

Focus on actionable insights for medicinal chemists.
"""
        
        response = self.llm.generate_response(prompt, temperature=0.3)
        recommendations = self._extract_recommendations(response)
        next_actions = self._suggest_next_actions(docking_df, 'docking')
        
        return AnalysisResult(
            summary=response,
            recommendations=recommendations,
            confidence_score=0.8,
            next_actions=next_actions,
            insights={'docking_stats': stats}
        )

    def analyze_generated_molecules(self, generated_smiles: List[str],
                                    generation_method: str = 'unknown') -> AnalysisResult:
        """Analyze generated molecules for quality and diversity"""
        stats = {
            'total_generated': len(generated_smiles),
            'unique_molecules': len(set(generated_smiles)),
            'avg_length': np.mean([len(smiles) for smiles in generated_smiles]),
            'generation_method': generation_method
        }
        
        sample_molecules = generated_smiles[:10] if len(generated_smiles) >= 10 else generated_smiles
        
        prompt = f"""
Analyze the following AI-generated molecules for drug discovery:

Generation Method: {generation_method}

Statistics:
- Total molecules generated: {stats['total_generated']}
- Unique molecules: {stats['unique_molecules']}
- Average SMILES length: {stats['avg_length']:.1f}

Sample Generated Molecules:
{chr(10).join([f"{i+1}. {smiles}" for i, smiles in enumerate(sample_molecules)])}

Please evaluate:
1. Chemical validity and drug-likeness
2. Structural diversity and novelty
3. Potential bioactivity based on known pharmacophores
4. Recommendations for filtering and optimization
5. Suggestions for improving the generation process

Provide actionable insights for the next iteration.
"""
        
        response = self.llm.generate_response(prompt, temperature=0.4)
        recommendations = self._extract_recommendations(response)
        next_actions = self._suggest_next_actions(generated_smiles, 'generation')
        
        return AnalysisResult(
            summary=response,
            recommendations=recommendations,
            confidence_score=0.7,
            next_actions=next_actions,
            insights={'generation_stats': stats}
        )

    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract actionable recommendations from LLM response"""
        recommendations = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '-', '•', '*']):
                clean_line = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '*']:
                    clean_line = clean_line.replace(prefix, '', 1).strip()
                
                if len(clean_line) > 10:
                    recommendations.append(clean_line)
        
        return recommendations[:10]

    def _suggest_next_actions(self, data: Any, analysis_type: str) -> List[str]:
        """Suggest concrete next actions based on analysis type"""
        if analysis_type == 'docking':
            return [
                "Select top 10-20 compounds for detailed analysis",
                "Perform molecular dynamics simulations",
                "Analyze binding mode and interactions",
                "Design analogs with improved binding",
                "Plan experimental validation"
            ]
        elif analysis_type == 'generation':
            return [
                "Filter molecules by drug-likeness criteria",
                "Remove duplicates and invalid structures",
                "Calculate molecular descriptors",
                "Perform virtual screening",
                "Generate additional diverse molecules"
            ]
        else:
            return [
                "Review analysis results",
                "Plan follow-up experiments",
                "Update models and parameters",
                "Iterate pipeline optimization"
            ]


if __name__ == "__main__":
    orchestrator = DrugDiscoveryOrchestrator('fallback')
    
    test_docking_df = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'c1ccccc1O'],
        'binding_affinity': [-8.5, -7.2, -9.1]
    })
    
    result = orchestrator.analyze_docking_results(test_docking_df, "EGFR kinase")
    print("Docking Analysis:")
    print(f"Summary: {result.summary[:200]}...")
    print(f"Recommendations: {result.recommendations[:3]}")
    print(f"Next actions: {result.next_actions[:3]}")
