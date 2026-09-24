import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dimension,
        context_length,
        num_heads,
        num_layers
    ):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dimension
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dimension
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension,
                    num_heads
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(
            embedding_dimension
        )

        # Convert hidden vectors → vocabulary scores
        self.output_layer = nn.Linear(
            embedding_dimension,
            vocab_size
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        # Token embeddings
        token_vectors = self.token_embedding(token_ids)

        # Position IDs
        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Position embeddings
        position_vectors = self.position_embedding(
            position_ids
        )

        # Combine token + position information
        x = token_vectors + position_vectors

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x, _ = block(x)

        # Final LayerNorm
        x = self.final_layer_norm(x)

        # Produce logits
        logits = self.output_layer(x)

        return logits


if __name__ == "__main__":

    vocab_size = 100
    embedding_dimension = 32
    context_length = 16
    num_heads = 4
    num_layers = 2

    model = GPTModel(
        vocab_size,
        embedding_dimension,
        context_length,
        num_heads,
        num_layers
    )

    # Example token IDs
    token_ids = torch.tensor([
        [5, 12, 23, 45]
    ])

    logits = model(token_ids)

    print("Input shape:")
    print(token_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)
    

if __name__ == "__main__":
    vocab_size = 53
    embedding_dimension = 32
    context_length = 4
    num_heads = 4
    num_layers = 2

    model = GPTModel(
        vocab_size,
        embedding_dimension,
        context_length,
        num_heads,
        num_layers
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total parameters:", total_parameters)   