import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import SimpleTokenizer
from dataset import TextDataset
from model import GPTModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
# --------------------------------
# 1. Load text
# --------------------------------

with open("data/text.txt", "r", encoding="utf-8") as file:
    text = file.read()


# --------------------------------
# 2. Tokenizer
# --------------------------------

tokenizer = SimpleTokenizer(text)

token_ids = tokenizer.encode(text)

vocab_size = len(tokenizer.token_to_id)

print("Vocabulary size:", vocab_size)



# 3. Create dataset
context_length = 16

full_dataset = TextDataset(
    token_ids,
    context_length
)

# Random train / validation split
train_size = int(len(full_dataset) * 0.9)
validation_size = len(full_dataset) - train_size

train_dataset, validation_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))



# 4. loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=2,
    shuffle=False
)


# --------------------------------
# 5. DataLoaders
# --------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=2,
    shuffle=False
)


# --------------------------------
# 6. Create GPT
# --------------------------------

embedding_dimension = 32
num_heads = 4   
num_layers = 2

model = GPTModel(
    vocab_size=vocab_size,
    embedding_dimension=embedding_dimension,
    context_length=context_length,
    num_heads=num_heads,  
    num_layers=num_layers
)
model = model.to(device)

# --------------------------------
# 7. Loss + Optimizer
# --------------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003
)


# --------------------------------
# 8. Checkpoint / Resume
# --------------------------------

epochs = 200
start_epoch = 0

patience = 10
epochs_without_improvement = 0

checkpoint_path = "checkpoint.pth"

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    print(
        f"Resuming training from epoch {start_epoch}"
    )

# --------------------------------
# 9. Training
# --------------------------------

# --------------------------------
# 9. Training
# --------------------------------

train_losses = []
validation_losses = []

best_validation_loss = float("inf")

patience = 10
epochs_without_improvement = 0

for epoch in range(start_epoch, epochs):

    # ----------------------------
    # Training
    # ----------------------------

    model.train()

    total_train_loss = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        logits = logits.view(
            -1,
            vocab_size
        )

        targets = targets.view(-1)

        loss = loss_function(
            logits,
            targets
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    train_losses.append(
        average_train_loss
    )


    # ----------------------------
    # Validation
    # ----------------------------

    model.eval()

    total_validation_loss = 0

    with torch.no_grad():

        for inputs, targets in validation_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            logits = logits.view(
                -1,
                vocab_size
            )

            targets = targets.view(-1)

            validation_loss = loss_function(
                logits,
                targets
            )

            total_validation_loss += (
                validation_loss.item()
            )

    average_validation_loss = (
        total_validation_loss /
        len(validation_loader)
    )

    validation_losses.append(
        average_validation_loss
    )


    # ----------------------------
    # Best model + Early stopping
    # ----------------------------

    if average_validation_loss < best_validation_loss:

        best_validation_loss = (
            average_validation_loss
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": average_validation_loss
            },
            "best_model.pth"
        )

        print("Best model saved!")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)"
        )

        if epochs_without_improvement >= patience:

            print("\nEarly stopping triggered!")

            break


    # ----------------------------
    # Print + checkpoint
    # ----------------------------

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": average_train_loss,
                "validation_loss": average_validation_loss,
                "token_to_id": tokenizer.token_to_id,
                "id_to_token": tokenizer.id_to_token
            },
            "checkpoint.pth"
        )
# --------------------------------
# 10. Save final model
# --------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token
    },
    "model.pth"
)

print("\nModel saved successfully!")

plt.plot(train_losses, label="Train Loss")
plt.plot(validation_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.show()