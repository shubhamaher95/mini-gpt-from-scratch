import torch
import torch.nn.functional as F

from tokenizer import SimpleTokenizer
from model import GPTModel


# --------------------------------
# 1. Load training text
# --------------------------------

with open("data/text.txt", "r", encoding="utf-8") as file:
    text = file.read()


# --------------------------------
# 2. Create tokenizer
# --------------------------------

tokenizer = SimpleTokenizer(text)

vocab_size = len(tokenizer.token_to_id)

context_length = 16
embedding_dimension = 32
num_heads = 4
num_layers = 2


# --------------------------------
# 3. Create model
# --------------------------------

model = GPTModel(
    vocab_size=vocab_size,
    embedding_dimension=embedding_dimension,
    context_length=context_length,
    num_heads=num_heads,
    num_layers=num_layers
)


# --------------------------------
# 4. Load trained weights
# --------------------------------

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# --------------------------------
# 5. Starting prompt
# --------------------------------

prompt = input("Enter your prompt: ")

token_ids = tokenizer.encode(
    prompt,
    add_eos=False
)
print("Prompt token IDs:", token_ids)
print("Prompt tokens:", tokenizer.decode(token_ids))

input_ids = torch.tensor(
    [token_ids],
    dtype=torch.long
)

# --------------------------------
# 6. Generate tokens
# --------------------------------

generated_tokens = token_ids.copy()

for _ in range(10):

    # Keep only the last context_length tokens
    input_context = generated_tokens[-context_length:]

    input_tensor = torch.tensor(
        [input_context],
        dtype=torch.long
    )

    # Get model predictions
    with torch.no_grad():
        logits = model(input_tensor)

    # Get predictions for the last position
    next_token_logits = logits[:, -1, :]

    # --------------------------------
    # Repetition penalty
    # --------------------------------

    repetition_penalty = 1.2

    for token_id in set(generated_tokens):
        next_token_logits[:, token_id] /= repetition_penalty

    # Prevent <UNK> from being generated
    unk_token_id = tokenizer.token_to_id["<UNK>"]

    next_token_logits[:, unk_token_id] = float("-inf")

    # --------------------------------
    # Temperature + Top-P
    # --------------------------------

    temperature = 0.8
    top_p = 0.9

    scaled_logits = next_token_logits / temperature

    probabilities = F.softmax(
        scaled_logits,
        dim=-1
    )

    # Sort probabilities
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True
    )

    # Calculate cumulative probabilities
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1
    )

    # Remove tokens outside Top-P
    remove_mask = cumulative_probabilities > top_p

    # Keep at least one token
    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
    remove_mask[:, 0] = False

    sorted_probabilities[remove_mask] = 0

    # Put probabilities back into original vocabulary order
    probabilities = torch.zeros_like(
        probabilities
    )

    probabilities.scatter_(
        1,
        sorted_indices,
        sorted_probabilities
    )

    # Normalize probabilities
    probabilities = probabilities / probabilities.sum(
        dim=-1,
        keepdim=True
    )

    # --------------------------------
    # Sample next token
    # --------------------------------

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    ).item()

    generated_tokens.append(next_token)

    # --------------------------------
    # Stop at EOS
    # --------------------------------

    eos_token_id = tokenizer.token_to_id["<EOS>"]

    if next_token == eos_token_id:
        break


# --------------------------------
# 7. Decode generated text
# --------------------------------

generated_text = tokenizer.decode(
    generated_tokens
)

print("\nGenerated text:")
print(generated_text)