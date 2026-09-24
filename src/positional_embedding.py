import torch
import torch.nn as nn


# Configuration
vocab_size = 10
embedding_dimension = 4
context_length = 8


# Token embedding
token_embedding = nn.Embedding(
    vocab_size,
    embedding_dimension
)


# Position embedding
position_embedding = nn.Embedding(
    context_length,
    embedding_dimension
)


# Example input
token_ids = torch.tensor([
    [2, 5, 7]
])


# Number of tokens in our sequence
sequence_length = token_ids.shape[1]


# Create position IDs
position_ids = torch.arange(sequence_length)


print("Token IDs:")
print(token_ids)

print("\nPosition IDs:")
print(position_ids)


# Get embeddings
token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(position_ids)


print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)


# Add token + position embeddings
input_vectors = token_vectors + position_vectors


print("\nFinal input vectors:")
print(input_vectors)

print("\nFinal shape:")
print(input_vectors.shape)