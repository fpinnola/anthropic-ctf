import torch
import torch.nn as nn

import string


vocab = " " + string.ascii_lowercase
vocab_size = len(vocab)
seq_length = 32


class ASCIIModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, seq_length: int):
        super(ASCIIModel, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.final = nn.Linear(seq_length * vocab_size, vocab_size * seq_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.seq_length * self.vocab_size)

        logits = self.final.forward(x)

        logits = logits.view(-1, self.seq_length, self.vocab_size)
        return logits

