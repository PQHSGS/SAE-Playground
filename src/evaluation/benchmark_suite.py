from typing import Dict, Any, List, Optional
import torch
from rich.table import Table
from rich.console import Console
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.core.base_dictionary import BaseDictionary
from src.evaluation.reconstruction.basic_metrics import compute_reconstruction_metrics
from src.evaluation.reconstruction.downstream_faithfulness import compute_ce_loss_recovery
from src.evaluation.feature_quality.feature_splitting import compute_feature_splitting_and_absorption
from src.evaluation.feature_quality.hierarchy_metrics import compute_hierarchy_and_monosemanticity
from src.evaluation.feature_quality.density_spectrum import compute_activation_density_spectrum


class SAEBenchmarkSuite:
    """
    SAEBench Comprehensive Evaluation Suite (OpenAI, DeepMind, Anthropic standard).
    Runs multi-dimensional evaluation covering:
    1. Reconstruction & Sparsity (NMSE, FVE, L0, L1, Cosine Similarity)
    2. Downstream Faithfulness (CE Loss Recovery, KL Divergence, Top-1 Agreement)
    3. Feature Splitting & Absorption (Mutual Coherence, Jaccard, Absorption Score)
    4. Monosemanticity & Hierarchy (Gini Sparseness, Skewness, Excess Kurtosis)
    5. Latent Density Spectrum (Dead, Ultra-rare, Common, Dense, Healthy Utilization)
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    @torch.no_grad()
    def evaluate(
        self,
        dictionary_model: BaseDictionary,
        activations: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        hook_point: Optional[str] = None,
        test_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results = {}

        # 1. Reconstruction Suite
        recon_metrics = compute_reconstruction_metrics(dictionary_model, activations, targets=targets)
        results["reconstruction"] = recon_metrics

        # 2. Density Spectrum Suite
        density_metrics = compute_activation_density_spectrum(dictionary_model, activations)
        results["density_spectrum"] = density_metrics

        # 3. Feature Splitting & Absorption Suite
        splitting_metrics = compute_feature_splitting_and_absorption(dictionary_model, activations)
        results["feature_splitting"] = splitting_metrics

        # 4. Hierarchy & Monosemanticity Suite
        hierarchy_metrics = compute_hierarchy_and_monosemanticity(dictionary_model, activations)
        results["monosemanticity"] = hierarchy_metrics

        # 5. Downstream Faithfulness (if model & test_texts provided)
        if model is not None and tokenizer is not None and hook_point is not None and test_texts:
            faithfulness_metrics = compute_ce_loss_recovery(
                model=model,
                tokenizer=tokenizer,
                dictionary_model=dictionary_model,
                hook_point=hook_point,
                test_texts=test_texts,
            )
            results["faithfulness"] = faithfulness_metrics

        return results

    def print_report(self, results: Dict[str, Any], title: str = "SAE Benchmark Evaluation Report"):
        table = Table(title=f"📊 [bold cyan]{title}[/bold cyan]", show_header=True, header_style="bold magenta")
        table.add_column("Evaluation Dimension", style="dim", width=28)
        table.add_column("Metric", style="bold yellow", width=32)
        table.add_column("Score / Value", justify="right", style="green")

        for category, metrics in results.items():
            first = True
            for k, v in metrics.items():
                cat_label = category.upper() if first else ""
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                table.add_row(cat_label, k.replace("_", " ").title(), val_str)
                first = False
            table.add_section()

        self.console.print(table)
