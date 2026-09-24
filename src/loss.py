import torch
import torch.nn as nn


# Example logits
logits = torch.tensor([
    [1.2, 0.8, 3.5, 0.4, 0.2]
])

# Correct answer is token ID 2
target = torch.tensor([2])


loss_function = nn.CrossEntropyLoss()

loss = loss_function(logits, target)

print("Loss:")
print(loss.item())