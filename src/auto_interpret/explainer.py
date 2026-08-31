from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.auto_interpret.prompts import EXPLANATION_SYSTEM_PROMPT, EXPLANATION_USER_PROMPT
from src.auto_interpret.sample_collector import ActivatingSnippet
from src.utils.logging import setup_logger

logger = setup_logger("explainer")


class FeatureExplainer:
    """
    100% Local LLM-powered explanation generator for dictionary features.
    Designed for local quantized instruct models (e.g. google/gemma-3-4b-it in 4-bit / 8-bit).
    Zero external API calls.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self._init_local_llm(model_name, load_in_4bit, load_in_8bit, torch_dtype)

    def _init_local_llm(self, model_name: str, load_in_4bit: bool, load_in_8bit: bool, torch_dtype: str):
        logger.info(f"Loading local auto-interpretation LLM: '{model_name}' (4-bit={load_in_4bit}, 8-bit={load_in_8bit})")
        try:
            dtype = torch.bfloat16 if torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": dtype,
            }
            if load_in_4bit:
                kwargs["load_in_4bit"] = True
            elif load_in_8bit:
                kwargs["load_in_8bit"] = True

            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model.eval()

            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=60,
                temperature=0.2,
                do_sample=False,
            )
            logger.info(f"Local LLM '{model_name}' initialized successfully on {self.device}.")
        except Exception as e:
            logger.warning(f"Could not load local LLM '{model_name}' ({e}). Falling back to heuristic feature explanation.")
            self.pipeline = None

    def _format_snippets(self, snippets: List[ActivatingSnippet]) -> str:
        formatted = []
        for i, s in enumerate(snippets[:8], 1):
            tok_strs = []
            for t in s.tokens:
                if t.activation_val > 0.1:
                    tok_strs.append(f"[{t.token_str} | {t.activation_val:.2f}]")
                else:
                    tok_strs.append(t.token_str)
            formatted.append(f"{i}. " + "".join(tok_strs))
        return "\n".join(formatted)

    def explain_feature(self, feature_id: int, snippets: List[ActivatingSnippet]) -> str:
        if not snippets:
            return "Dead / Inactive feature with no observed activations."

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
                    formatted_prompt = f"{EXPLANATION_SYSTEM_PROMPT}\n\n{prompt}\nExplanation:"

                out = self.pipeline(formatted_prompt)
                gen_text = out[0]["generated_text"]
                if formatted_prompt in gen_text:
                    gen_text = gen_text.replace(formatted_prompt, "").strip()
                return gen_text.split("\n")[0].strip()
            except Exception as e:
                logger.warning(f"Local LLM explanation generation failed: {e}")

        # Fast heuristic extraction for offline/unit test execution
        top_toks = set()
        for s in snippets[:5]:
            for t in s.tokens:
                if t.activation_val > 0.5 * s.max_activation:
                    top_toks.add(t.token_str.strip())
        return f"Activates on tokens/concepts related to: {list(top_toks)[:8]}"
