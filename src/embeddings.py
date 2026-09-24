import torch
import torch.nn as nn


# Our vocabulary size
vocab_size = 10

# Number of values used to represent each token
embedding_dimension = 4


# Create embedding layer
embedding = nn.Embedding(
    vocab_size,
    embedding_dimension
)


# Example token IDs
token_ids = torch.tensor([2, 5, 7])


# Convert token IDs into vectors
vectors = embedding(token_ids)


print("Token IDs:")
print(token_ids)

print("\nEmbeddings:")
print(vectors)

print("\nEmbedding shape:")
print(vectors.shape)