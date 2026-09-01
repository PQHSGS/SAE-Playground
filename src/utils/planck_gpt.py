from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standard non-parametric RMSNorm for NanoChat/PlanckGPT."""
    return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)


class PlanckMQAAttention(nn.Module):
    """
    Multi-Query Attention (MQA) for PlanckGPT:
    Single Key and Value head shared across all Query heads with SDPA.
    """
    def __init__(self, d_model: int = 896, num_heads: int = 7, head_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)               # (B, 1, S, D)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)               # (B, 1, S, D)

        # Expand K and V across query heads for SDPA
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = (attention_mask[:, None, None, :] == 0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(attention_mask is None),
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class PlanckBlock(nn.Module):
    def __init__(self, d_model: int = 896, num_heads: int = 7, head_dim: int = 128, d_ff: int = 3584):
        super().__init__()
        self.attn = PlanckMQAAttention(d_model=d_model, num_heads=num_heads, head_dim=head_dim)
        self.ffn1 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-RMSNorm Attention & MLP
        attn_out = self.attn(rms_norm(x), attention_mask=attention_mask)
        x = x + attn_out
        mlp_out = self.ffn2(F.gelu(self.ffn1(rms_norm(x))))
        x = x + mlp_out
        return x


class PlanckGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 896,
        num_layers: int = 14,
        num_heads: int = 7,
        head_dim: int = 128,
        d_ff: int = 3584,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.resid_lambdas = nn.Parameter(torch.ones(num_layers))
        self.x0_lambdas = nn.Parameter(torch.ones(num_layers))
        
        self.transformer = nn.ModuleList([
            PlanckBlock(d_model=d_model, num_heads=num_heads, head_dim=head_dim, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x = rms_norm(self.embedding(input_ids))
        x0 = x

        for i, block in enumerate(self.transformer):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, attention_mask=attention_mask)

        logits = self.output(rms_norm(x))
        return logits
