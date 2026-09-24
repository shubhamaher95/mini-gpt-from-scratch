import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dimension, num_heads):
        super().__init__()

        assert embedding_dimension % num_heads == 0

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads

        self.WQ = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.WK = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.WV = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.output_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension
        )

    def forward(self, X):

        batch_size, sequence_length, _ = X.shape

        # Create Q, K, V
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        # Split into multiple heads
        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dimension)

        # Causal mask
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=X.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Attention weights
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        output = attention_weights @ V

        # Combine heads
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dimension
        )

        # Final projection
        output = self.output_projection(output)

        return output, attention_weights