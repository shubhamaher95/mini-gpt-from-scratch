import torch
import math


# Three tokens
# Each token has 4 dimensions
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # Token 1
    [0.0, 2.0, 0.0, 2.0],   # Token 2
    [1.0, 1.0, 1.0, 1.0]    # Token 3
])


print("Input X:")
print(X)


# Create Q, K, V
# For now, we use X directly.
Q = X
K = X
V = X


# Calculate attention scores
scores = Q @ K.T


print("\nAttention scores:")
print(scores)


# Scale scores
d_k = K.shape[-1]

scaled_scores = scores / math.sqrt(d_k)


print("\nScaled scores:")
print(scaled_scores)


# Convert scores into probabilities
attention_weights = torch.softmax(
    scaled_scores,
    dim=-1
)


print("\nAttention weights:")
print(attention_weights)


# Weighted sum of values
output = attention_weights @ V


print("\nAttention output:")
print(output)
