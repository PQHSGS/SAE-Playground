# Interpret Playground ⚡

A unified, modular research playground for constructing, training, evaluating, auto-interpreting, and interactively exploring **Sparse Autoencoders (SAEs)**, **Transcoders**, and **Crosscoders** for arbitrary Large Language Models (including standard models like Gemma 3, LLaMA 3, Qwen 2.5, and research architectures like PlanckGPT).

---

## 🌟 Key Features

1. **State-of-the-Art Dictionary Architectures**:
   - **TopK SAE** *(OpenAI 2024)*: Exact per-token top-$k$ sparsity, eliminating shrinkage without $L_1$ penalty tuning.
   - **BatchTopK SAE**: Global top $B \times k$ selection across token batches for adaptive token capacity.
   - **JumpReLU SAE** *(Google DeepMind 2024)*: Learned per-feature activation thresholds with Straight-Through Estimators (STE).
   - **Gated SAE** *(Rajamanoharan et al. 2024)*: Decoupled gating and magnitude linear paths.
   - **Matryoshka / Multi-Scale SAE** *(Bussmann et al. 2025)*: Nested dictionary representations ($[512, 1024, 2048, 4096]$) trained with joint multi-prefix loss.
   - **Matching Pursuit SAE**: Iterative residual projection (tied OMP).
   - **TreeSAE & SASA**: Hierarchical coarse-to-fine concept routing and sparsity-adaptive dynamic $k(x)$.
   - **Skip-Transcoder**: Residual-preserving transcoder with dedicated linear skip bypass $W_{\text{skip}} x$.
   - **Multi-Layer Crosscoder** *(Anthropic 2024)*: Shared latent space across $L$ transformer layers.

2. **Universal Model Loading & Dynamic Hooking**:
   - Compatible with any Hugging Face model (`AutoModelForCausalLM` with `trust_remote_code=True`), local weights, or custom research architectures.
   - Hook into **Main Residual Stream**, **MLP Sublayers**, or **Attention Outputs** by dot-path.
   - Automatic Unembedding Matrix $W_U$ extraction for **Logit Lens / Direct Logit Attribution**.

3. **100% Local Auto-Interpretation**:
   - Quantized Instruct LLMs (e.g. `google/gemma-3-4b-it` in 4-bit / 8-bit). Zero external API calls.
   - Token-level activation heatmaps & faithfulness simulation scoring.
   - **Edge Attribution Patching (EAP-SAE)** for feature circuit discovery.

4. **Neuronpedia-Style Interactive Dashboard**:
   - Web UI for feature search, token heatmaps, direct logit attribution, and real-time causal feature steering.

---

## 🚀 Quick Start

### 1. Environment Setup
Activate your conda environment:
```bash
conda activate sae_circuit
# Optional dependencies for Dashboard
pip install fastapi uvicorn
```

---

## 📖 Usage Guide

All execution in Interpret Playground is driven by single YAML configuration files.

### 1. Training

#### Train Full Architecture SAE on Gemma 3 270M (All 18 Depth Layers Simultaneously):
```bash
python scripts/train.py --config configs/gemma3_270m_all_layers.yaml
```

#### Train Single-Layer SAE on Gemma 3 270M (Layer 9):
```bash
python scripts/train.py --config configs/gemma3_270m_single_layer.yaml
```

#### Train Full Architecture SAE on PlanckGPT (All 14 Depth Layers Simultaneously):
```bash
python scripts/train.py --config configs/planckgpt_all_layers.yaml
```

#### Train Single-Layer SAE on PlanckGPT (Layer 7):
```bash
python scripts/train.py --config configs/planckgpt_single_layer.yaml
```

#### Train TopK SAE on GPT-2:
```bash
python scripts/train.py --config configs/gpt2_topk.yaml
```

#### Train Skip-Transcoder:
```bash
python scripts/train.py --config configs/skip_transcoder.yaml
```

#### Train Multi-Layer Crosscoder:
```bash
python scripts/train.py --config configs/crosscoder.yaml
```

---

### 2. Benchmarking & Evaluation

Evaluate reconstruction error (MSE / NMSE), $L_0$, explained variance, dead feature percentage, and Cross-Entropy loss recovery automatically from your config:
```bash
python scripts/evaluate.py --config configs/gemma3_270m_all_layers.yaml
```

---

### 3. Interactive Neuronpedia Dashboard

Launch the local web UI (feature search, token heatmaps, direct logit attribution, and feature steering):
```bash
python scripts/launch_dashboard.py --config configs/gemma3_270m_all_layers.yaml --port 8000
```
Open `http://localhost:8000` in your browser.

---

### 4. Causal Feature Steering

Test steering features during text generation:
```bash
python scripts/steer.py --config configs/gemma3_270m_all_layers.yaml --feature_id 42 --alpha 15.0
```

---

### 5. Local Auto-Interpretation

Extract max-activating snippets and generate local explanations with 0 external API dependencies:
```bash
python scripts/run_auto_interp.py --config configs/auto_interp.yaml --num_features 20
```

---

### 6. Transcoder Circuit & Graph Discovery

Extract mechanistic circuits and causal transcoder graphs:
```bash
python scripts/run_circuits.py --config configs/circuits.yaml
```

---

## 📂 Repository Structure

```
.
├── configs/                          # Clean self-contained YAML experiment configs
│   ├── gemma3_270m_all_layers.yaml   # Gemma 3 270M (All 18 Depth Layers Simultaneously)
│   ├── gemma3_270m_single_layer.yaml # Gemma 3 270M (Single Layer 9)
│   ├── planckgpt_all_layers.yaml     # PlanckGPT (All 14 Depth Layers Simultaneously)
│   ├── planckgpt_single_layer.yaml   # PlanckGPT (Single Layer 7)
│   ├── gpt2_topk.yaml                # GPT-2 TopK SAE
│   ├── skip_transcoder.yaml          # Skip-Transcoder
│   ├── crosscoder.yaml               # Multi-Layer Crosscoder
│   ├── auto_interp.yaml              # Local auto-interpretation settings
│   └── circuits.yaml                 # Edge Attribution Patching (EAP-SAE) & Transcoder Graphs
├── src/
│   ├── core/                        # BaseDictionary, HookManager, ActivationBuffer, Configs
│   ├── architectures/               # SAEs, Transcoders, Crosscoders & Factory Registry
│   ├── training/                    # Trainer, Loss functions, ConstrainedAdamW, Schedulers
│   ├── evaluation/                  # SAEBench (Reconstruction, Faithfulness, Splitting, Hierarchy)
│   ├── auto_interpret/              # Max-activating token collector, Local Explainer, Simulator
│   ├── circuits/                    # EAP-SAE, Transcoder Circuits, Feature Steering
│   ├── dashboard/                   # FastAPI backend + Modern UI
│   └── utils/                       # Universal HF loaders, Logit Lens (W_U), IO, Rich Logger
├── scripts/                         # CLI scripts (train, evaluate, steer, dashboard, auto-interp, circuits)
├── tests/                           # Automated pytest suite (24 tests)
├── data/                            # Data caches, auto-interp results, circuit graphs
└── checkpoints/                     # Saved model weights and metadata
```

