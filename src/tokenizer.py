import re


class SimpleTokenizer:

    def __init__(self, text):

        # Convert text to lowercase and split into words
        words = re.findall(
            r"\w+|[^\w\s]",
            text.lower()
        )

        # Get unique words
        unique_words = sorted(set(words))

        # Special tokens
        self.token_to_id = {
            "<UNK>": 0,
            "<PAD>": 1,
            "<EOS>": 2
        }

        # Assign an ID to every word
        for word in unique_words:
            self.token_to_id[word] = len(
                self.token_to_id
            )

        # Reverse dictionary: ID -> word
        self.id_to_token = {
            token_id: token
            for token, token_id
            in self.token_to_id.items()
        }

    def encode(self, text, add_eos=True):

        words = text.lower().split()

        token_ids = [
            self.token_to_id.get(
                word,
                self.token_to_id["<UNK>"]
            )
            for word in words
        ]

        # Add End Of Sequence token
        # Add EOS only when requested
        if add_eos:
          token_ids.append(
        self.token_to_id["<EOS>"]
    )

        return token_ids

    def decode(self, token_ids):

        return " ".join(
            self.id_to_token[token_id]
            for token_id in token_ids
        )

# Test our tokenizer
if __name__ == "__main__":

    text = """
    I love AI.
    I love Python.
    Python is powerful.
    """

    tokenizer = SimpleTokenizer(text)

    print("Vocabulary:")
    print(tokenizer.token_to_id)

    sentence = "I love Python"

    encoded = tokenizer.encode(sentence)

    print("\nEncoded:")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded:")
    print(decoded)