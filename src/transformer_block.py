import torch
import torch.nn as nn

from self_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()

        hidden_dimension = embedding_dimension * 4

        self.network = nn.Sequential(
    nn.Linear(embedding_dimension, hidden_dimension),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dimension, embedding_dimension)
)

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadAttention(
            embedding_dimension,
            num_heads
        )
        self.dropout_1 = nn.Dropout(0.1)

        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.dropout_2 = nn.Dropout(0.1)
        self.feed_forward = FeedForward(embedding_dimension)

    def forward(self, x):

        # Attention + Residual Connection
        normalized_x = self.layer_norm_1(x)

        attention_output, attention_weights = self.attention(
            normalized_x
        )

        x = x + self.dropout_1(attention_output)

        # Feed-Forward + Residual Connection
        normalized_x = self.layer_norm_2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        x = x + self.dropout_2(feed_forward_output)

        return x, attention_weights


if __name__ == "__main__":

    embedding_dimension = 8
    num_heads = 2

    x = torch.randn(1, 4, embedding_dimension)

    transformer_block = TransformerBlock(
        embedding_dimension,
        num_heads
    )

    output, attention_weights = transformer_block(x)

    print("Input shape:")
    print(x.shape)

    print("\nAttention weights shape:")
    print(attention_weights.shape)

    print("\nOutput shape:")
    print(output.shape)