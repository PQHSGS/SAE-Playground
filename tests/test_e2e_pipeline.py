import os
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.architectures.sae.topk_sae import TopKSAE
from src.core.activation_buffer import ActivationBuffer
from src.core.config import TrainingConfig
from src.training.trainer import DictionaryTrainer
from src.evaluation.metrics import compute_reconstruction_metrics
from src.evaluation.feature_stats import compute_feature_statistics
from src.circuits.steering import FeatureSteeringEngine
from src.utils.hf_helpers import get_unembedding_weights, compute_direct_logit_attribution


def test_end_to_end_sae_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load small base model
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    hook_point = "transformer.h.2"
    d_in = model.config.n_embd  # 768
    d_sae = 256  # small test dictionary
    k = 16

    # 2. Build TopK SAE
    dict_model = TopKSAE(d_in=d_in, d_sae=d_sae, k=k).to(device)

    # 3. Setup Activation Buffer
    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=[hook_point],
        batch_size=64,
        buffer_size=256,
        model_batch_size=2,
        context_length=64,
        device=device,
    )

    # 4. Train for 5 steps in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        train_cfg = TrainingConfig(
            batch_size=64,
            learning_rate=1e-3,
            total_steps=5,
            checkpoint_steps=5,
            output_dir=tmpdir,
            wandb_project=None,
        )

        trainer = DictionaryTrainer(
            dictionary_model=dict_model,
            activation_buffer=buffer,
            config=train_cfg,
            device=device,
        )
        trainer.train()

        # Check saved checkpoint
        ckpt_path = os.path.join(tmpdir, "step_5")
        assert os.path.exists(os.path.join(ckpt_path, "config.json"))
        assert os.path.exists(os.path.join(ckpt_path, "model.safetensors"))

        # 5. Evaluate metrics
        eval_acts = buffer.next_batch()
        metrics = compute_reconstruction_metrics(dict_model, eval_acts)
        assert "nmse" in metrics
        assert metrics["l0"] > 0

        stats = compute_feature_statistics(dict_model, eval_acts)
        assert "dead_features_count" in stats

        # 6. Feature steering
        steering = FeatureSteeringEngine(
            model=model,
            tokenizer=tokenizer,
            dictionary_model=dict_model,
            hook_point=hook_point,
        )
        gen = steering.generate_with_steering(
            prompt="Machine learning is",
            steered_features={0: 5.0},
            max_new_tokens=10,
        )
        assert len(gen) > len("Machine learning is")

        # 7. Logit lens
        w_u = get_unembedding_weights(model)
        w_dec = dict_model.get_decoder_weights()
        promoted, suppressed = compute_direct_logit_attribution(
            decoder_vector=w_dec[0],
            unembedding_weights=w_u,
            tokenizer=tokenizer,
            top_k=5,
        )
        assert len(promoted) == 5
        assert len(suppressed) == 5
