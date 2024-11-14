import torch
import math
from torch import nn, Tensor
from dataclasses import dataclass
from typing import List


@dataclass
class LanguageConfig:
    model_name: str = "vinai/bartpho-word"
    max_length: int = 64
    padding: str = 'max_length'
    truncation: bool = True
    return_token_type_ids: bool = False
    hidden_size: int = 1024
    self_attn_heads: int = 8
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    layer_norm_eps: float = 1e-6


class CrossAttention(nn.Module):
    def __init__(self, encoder_hidden_size, num_heads, hidden_size, dropout) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(
            hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)

        self.projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states=None,
                encoder_hidden_states=None,
                ):

        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(
            self.value(encoder_hidden_states))
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs_dropped = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)

        output = context_layer.view(*new_context_layer_shape)
        return self.resid_dropout(self.projection(output))


class MLP(nn.Module):
    def __init__(self, config: LanguageConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.fc2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CrossAugmentation(nn.Module):
    def __init__(self, attn_layer, mlp_layer, norm_layer):
        super().__init__()
        self.attn = attn_layer
        self.mlp = mlp_layer
        self.norm = norm_layer

    def forward(self, query: Tensor, encode_hidden_states: List[Tensor]) -> Tensor:
        for encode_hidden_state in encode_hidden_states:
            attn_output = self.attn(query, encode_hidden_state)
            query = self.norm(query + attn_output)
            mlp_output = self.mlp(query)
            query = self.norm(query + mlp_output)
        return query
