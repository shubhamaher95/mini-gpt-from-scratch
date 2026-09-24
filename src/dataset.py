import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):

        input_ids = self.token_ids[
            index:index + self.context_length
        ]

        target_ids = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )


# Test
if __name__ == "__main__":

    token_ids = [10, 20, 30, 40, 50, 60]

    dataset = TextDataset(
        token_ids,
        context_length=3
    )

    print("Number of samples:", len(dataset))

    for i in range(len(dataset)):
        inputs, targets = dataset[i]

        print("\nInput :", inputs.tolist())
        print("Target:", targets.tolist())