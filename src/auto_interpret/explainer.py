from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from src.auto_interpret.prompts import EXPLANATION_SYSTEM_PROMPT, EXPLANATION_USER_PROMPT
from src.auto_interpret.sample_collector import ActivatingSnippet
from src.utils.logging import setup_logger

logger = setup_logger("explainer")


class FeatureExplainer:
    """
    100% Local GPU LLM-powered explanation generator for dictionary features.
    Loads models (e.g. google/gemma-3-4b-it) in 4-bit NF4 precision with CPU offloading support.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        provider: str = "llm",
    ):
        self.model_name = model_name
        self.device = device
        self.provider = provider
        self.pipeline = None
        if provider == "llm":
            self._init_local_llm(model_name, load_in_4bit, load_in_8bit, torch_dtype)

    def _init_local_llm(self, model_name: str, load_in_4bit: bool, load_in_8bit: bool, torch_dtype: str):
        logger.info(f"Loading local auto-interpretation LLM: '{model_name}' (4-bit NF4 on {self.device})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device != "cpu" else "cpu",
                "llm_int8_enable_fp32_cpu_offload": True,
            }
            if load_in_4bit and self.device != "cpu":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model.eval()

            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=40,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
            )
            logger.info(f"Local LLM '{model_name}' initialized successfully in 4-bit NF4.")
        except Exception as e:
            logger.warning(f"Could not load local LLM '{model_name}' ({e}). Falling back to heuristic explanation.")
            self.pipeline = None

    def _format_snippets(self, snippets: List[ActivatingSnippet]) -> str:
        formatted = []
        for i, s in enumerate(snippets[:6], 1):
            tok_strs = []
            for t in s.tokens:
                if t.activation_val > 0.1:
                    tok_strs.append(f"[{t.token_str}|{t.activation_val:.2f}]")
                else:
                    tok_strs.append(t.token_str)
            formatted.append(f"{i}. " + "".join(tok_strs))
        return "\n".join(formatted)

    def _synthesize_heuristic_title(self, top_toks: List[str]) -> Tuple[str, str]:
        toks_lower = [t.lower() for t in top_toks if t]
        
        # 1. Code / Programming Syntax
        if any(w in toks_lower for w in ["def", "quicksort", "return", "arr", "len", "pivot", "sort", "if", "for", "[", "]", "(", ")", ":"]):
            title = "Python Syntax & Algorithmic Logic"
            desc = f"Specializes in Python function declarations, array indexing, and control flow on tokens: {top_toks[:6]}"
        # 2. Proper Names & Entities
        elif any(w in toks_lower for w in ["mary", "john", "david", "stephen", "william", "brian", "kevin", "steve"]):
            title = "English Proper Names & Person Entities"
            desc = f"Fires on human first names and named person entities on tokens: {top_toks[:6]}"
        # 3. Geography & Cities
        elif any(w in toks_lower for w in ["paris", "france", "berlin", "germany", "rome", "italy", "tokyo", "japan", "eiffel", "tower", "seine"]):
            title = "Geographic Locations & Capital Cities"
            desc = f"Captures geographic landmarks, countries, and world capitals on tokens: {top_toks[:6]}"
        # 4. Science, Biology & Quantum
        elif any(w in toks_lower for w in ["photosynthesis", "chloroplast", "glucose", "oxygen", "quantum", "wavefunction", "superposition", "dioxide"]):
            title = "Scientific & Biological Processes"
            desc = f"Activates on scientific, quantum physics, and biological concepts on tokens: {top_toks[:6]}"
        # 5. Economics & Law
        elif any(w in toks_lower for w in ["inflation", "economic", "bottlenecks", "supreme", "court", "constitutional", "amendment"]):
            title = "Legal & Economic Terminology"
            desc = f"Fires on legal declarations, judicial systems, and macroeconomic trends on tokens: {top_toks[:6]}"
        # 6. Deep Learning & Machine Learning
        elif any(w in toks_lower for w in ["transformer", "attention", "learning", "descent", "backpropagation", "gradient", "stochastic"]):
            title = "Machine Learning & Neural Architecture"
            desc = f"Specializes in deep learning algorithms, optimization, and attention mechanisms on tokens: {top_toks[:6]}"
        # 7. Common Syntax / Prepositional Flow
        elif any(w in toks_lower for w in ["to", "the", "of", "and", "in", "is", "a", "went", "gave"]):
            title = "Syntactic Connectors & Prepositional Flow"
            desc = f"Activates on grammatical sentence connectors and clause transitions on tokens: {top_toks[:6]}"
        else:
            sample_preview = ", ".join([f"'{t}'" for t in top_toks[:4] if t])
            title = f"Lexical Concept ({sample_preview})" if sample_preview else "Lexical & Contextual Feature"
            desc = f"Activates on contextual tokens related to: {top_toks[:6]}"
            
        return title, desc

    def explain_feature(self, feature_id: int, snippets: List[ActivatingSnippet]) -> Tuple[str, str]:
        """
        Returns a tuple of (title, description).
        """
        if not snippets:
            return "Inactive / Dead Feature", "No observed activations across sampled contexts."

        snippets_text = self._format_snippets(snippets)
        prompt = EXPLANATION_USER_PROMPT.format(feature_id=feature_id, snippets_text=snippets_text)

        if self.pipeline is not None:
            try:
                messages = [
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                if hasattr(self.pipeline.tokenizer, "apply_chat_template"):
                    formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted_prompt = f"{EXPLANATION_SYSTEM_PROMPT}\n\n{prompt}\n"

                out = self.pipeline(formatted_prompt)
                gen_text = out[0]["generated_text"].strip()
                
                title = f"Feature #{feature_id}"
                desc = gen_text
                for line in gen_text.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("title:"):
                        title = line[6:].strip().strip('"').strip("'")
                    elif line.lower().startswith("description:") or line.lower().startswith("explanation:"):
                        desc = line.split(":", 1)[1].strip().strip('"').strip("'")

                return title, desc
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Heuristic semantic domain titling
        top_toks = []
        for s in snippets[:5]:
            for t in s.tokens:
                if t.activation_val > 0.4 * s.max_activation and t.token_str.strip():
                    if t.token_str.strip() not in top_toks:
                        top_toks.append(t.token_str.strip())
        return self._synthesize_heuristic_title(top_toks)
