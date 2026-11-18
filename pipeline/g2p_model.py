"""Dense Bengali character-to-phoneme encoder/decoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS, UNK = range(4)


def encode_word(word, vocab):
    return [BOS, *(vocab.get(char, UNK) for char in word), EOS]


class GraphemeToPhone(nn.Module):
    def __init__(self, input_size, output_size, width=160):
        super().__init__()
        self.source_embedding = nn.Embedding(input_size, width, padding_idx=PAD)
        self.target_embedding = nn.Embedding(output_size, width, padding_idx=PAD)
        self.encoder = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(width, width, batch_first=True)
        self.initial = nn.Linear(width, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.join = nn.Linear(width * 2, width)
        self.projection_in = nn.Linear(width, width)
        self.projection_out = nn.Linear(width, width)
        self.normalization = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_size)
        self.dropout = nn.Dropout(0.10)

    def encode(self, source):
        mask = source.ne(PAD)
        # Right padding must not influence the backward encoder state for a word.
        lengths = mask.sum(1).to("cpu")
        embedded = self.dropout(self.source_embedding(source))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_memory, state = self.encoder(packed)
        memory, _ = nn.utils.rnn.pad_packed_sequence(
            packed_memory, batch_first=True, total_length=source.shape[1]
        )
        hidden = self.initial(torch.cat((state[0], state[1]), dim=-1)).tanh().unsqueeze(0)
        return memory, self.key(memory).transpose(1, 2), mask, hidden

    def decode(self, target, memory, keys, mask, hidden):
        states, hidden = self.decoder(self.dropout(self.target_embedding(target)), hidden)
        scores = self.query(states) @ keys / math.sqrt(states.shape[-1])
        weights = scores.masked_fill(~mask[:, None, :], -torch.inf).softmax(dim=-1)
        features = F.gelu(self.join(torch.cat((states, weights @ memory), dim=-1)))
        update = self.projection_out(F.gelu(self.projection_in(features)))
        features = self.normalization(features + self.dropout(update))
        return self.output(self.dropout(features)), hidden

    def forward(self, source, target):
        logits, _ = self.decode(target, *self.encode(source))
        return logits

    @torch.inference_mode()
    def predict(self, source, max_steps):
        memory, keys, mask, hidden = self.encode(source)
        token = source.new_full((source.shape[0], 1), BOS)
        finished = source.new_zeros(source.shape[0], dtype=torch.bool)
        sequence = []
        for _ in range(max_steps):
            logits, hidden = self.decode(token, memory, keys, mask, hidden)
            logits[..., :EOS] = -torch.inf
            logits[..., UNK] = -torch.inf
            token = logits.argmax(-1)
            token[finished] = EOS
            sequence.append(token)
            finished |= token[:, 0].eq(EOS)
            if finished.all():
                break
        return torch.cat(sequence, dim=-1)
