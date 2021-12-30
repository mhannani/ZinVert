import torch
import torch.nn as nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    """
    The attention model
    """

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.attention_hidden_vector = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.attention_scoring_fn = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_length = encoder_outputs.shape[0]

        hidden = hidden.repeat(src_length, 1, 1)

        attention_hidden = torch.tanh(self.attention_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))

        attention_scoring_vector = self.attention_scoring_fn(attention_hidden).squeeze(2)

        attention_scoring_vector = attention_scoring_vector.permute(1, 0)

        # output the probability distribution
        return softmax(attention_scoring_vector, dim=1)
