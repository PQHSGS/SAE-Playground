import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanckMQAAttention(nn.Module):
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)                     # (B, H, S, S)
        
        # Causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
                scores = scores + pad_mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)                                                           # (B, H, S, D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class PlanckBlock(nn.Module):
    def __init__(self, d_model: int = 896, num_heads: int = 7, head_dim: int = 128, d_ff: int = 3584):
        super().__init__()
        self.attn = PlanckMQAAttention(d_model=d_model, num_heads=num_heads, head_dim=head_dim)
        self.ffn1 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, attention_mask=attention_mask)
        x = x + attn_out
        mlp_out = self.ffn2(F.gelu(self.ffn1(x)))
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
        x = self.embedding(input_ids)
        x0 = x

        for i, block in enumerate(self.transformer):
            x_next = block(x, attention_mask=attention_mask)
            x = self.resid_lambdas[i] * x_next + self.x0_lambdas[i] * x0

        logits = self.output(x)
        return logits
