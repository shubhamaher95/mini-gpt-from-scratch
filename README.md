# Mini GPT From Scratch

A small GPT-style Large Language Model built from scratch using Python and PyTorch.

This project is an educational implementation of the core components used in a GPT-style language model. The goal is to understand how an LLM works internally, from converting text into tokens to generating new text.

## 🧠 Project Architecture

```text
                 Input Text
                     │
                     ▼
                Tokenizer
                     │
                     ▼
                 Token IDs
                     │
                     ▼
        Token + Positional Embeddings
                     │
                     ▼
           Multi-Head Self-Attention
                     │
                     ▼
             Feed Forward Network
                     │
                     ▼
             Transformer Blocks
                     │
                     ▼
              Layer Normalization
                     │
                     ▼
                Output Layer
                     │
                     ▼
                   Logits
                     │
                     ▼
             Next Token Prediction

📁 Project Structure
mini-gpt-from-scratch/
│
├── data/
│   └── text.txt
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── positional_embedding.py
│   ├── attention.py
│   ├── self_attention.py
│   ├── multi_head_attention.py
│   ├── transformer_block.py
│   ├── model.py
│   ├── loss.py
│   ├── train_demo.py
│   ├── train.py
│   └── generate.py
│
├── .gitignore
├── README.md
└── requirements.txt

🧩 How It Works
1. Tokenization

The input text is converted into smaller tokens.

Example:

Input:
I love artificial intelligence

Tokens:
I
love
artificial
intelligence

These tokens are then converted into numerical token IDs.

2. Token Embeddings

Each token ID is converted into a vector representation.

Token ID
   ↓
Embedding Layer
   ↓
Vector
3. Positional Embeddings

The model also needs to understand the position of each token.

For example:

I       → Position 0
love    → Position 1
AI      → Position 2

Token embeddings and positional embeddings are combined before entering the Transformer.

4. Self-Attention

Self-attention allows each token to look at previous tokens and understand their relationships.

The model uses causal masking so that a token cannot see future tokens during training.

5. Multi-Head Attention

Multiple attention heads allow the model to learn different relationships between tokens.

Input
  │
  ├── Attention Head 1
  ├── Attention Head 2
  ├── Attention Head 3
  └── Attention Head 4
          │
          ▼
      Combined Output
6. Transformer Block

Each Transformer block contains:

Multi-Head Attention
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Feed Forward Network
        ↓
Residual Connection
        ↓
Layer Normalization

The project currently uses 2 Transformer blocks.

7. Next Token Prediction

The model predicts the next token based on the previous tokens.

Example:

Input:
I love

Prediction:
artificial

Then the generated token is added to the sequence and the model predicts the next token.

I love → artificial
I love artificial → intelligence

This process continues until the generation stops.
                     │
                     ▼
              Generated Text
