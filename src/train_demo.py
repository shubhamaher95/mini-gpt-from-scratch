import torch
import torch.nn as nn


# -------------------------
# 1. Create training data
# -------------------------

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])


# -------------------------
# 2. Create a small model
# -------------------------

model = nn.Linear(1, 1)


# -------------------------
# 3. Loss function
# -------------------------

loss_function = nn.MSELoss()


# -------------------------
# 4. Optimizer
# -------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# -------------------------
# 5. Training loop
# -------------------------

for epoch in range(1000):

    # Forward pass
    predictions = model(X)

    # Calculate loss
    loss = loss_function(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate new gradients
    loss.backward()

    # Update model weights
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.4f}"
        )


# -------------------------
# 6. Test the model
# -------------------------

test_input = torch.tensor([[5.0]])

prediction = model(test_input)

print("\nPrediction for 5:")
print(prediction.item())