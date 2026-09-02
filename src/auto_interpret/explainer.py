from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._init_local_llm(model_name, load_in_4bit, load_in_8bit, torch_dtype)

    def _init_local_llm(self, model_name: str, load_in_4bit: bool, load_in_8bit: bool, torch_dtype: str):
        from transformers import BitsAndBytesConfig

        logger.info(f"Loading local auto-interpretation LLM: '{model_name}' on {self.device} (4-bit NF4)...")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device,
            "attn_implementation": "eager",
        }
        if load_in_4bit and self.device == "cuda":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        logger.info(f"Local quantized LLM '{model_name}' ready on {self.device}.")

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

    def explain_feature(self, feature_id: int, snippets: List[ActivatingSnippet]) -> Tuple[str, str]:
        """
        Generates conceptual title and plain-English explanation using local quantized LLM.
        """
        if not snippets:
            return f"Feature #{feature_id}", "No activations observed in sampled contexts."

        snippets_text = self._format_snippets(snippets)
        user_prompt = EXPLANATION_USER_PROMPT.format(feature_id=feature_id, snippets_text=snippets_text)

        messages = [
            {"role": "user", "content": f"{EXPLANATION_SYSTEM_PROMPT}\n\n{user_prompt}"}
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                ids,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = outputs[0, ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        title = f"Feature #{feature_id}"
        desc = gen_text
        for line in gen_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("title:"):
                title = line[6:].strip().strip('"').strip("'")
            elif line.lower().startswith("description:") or line.lower().startswith("explanation:"):
                desc = line.split(":", 1)[1].strip().strip('"').strip("'")

        return title, desc
