import random
from src.data.config import *
import torch
import torch.nn as nn


class OneStepDecoder(nn.Module):
    """
    The OneStepDecoder model.
    """

    def __init__(self, input_output_dim, embedding_dim, hidden_dim, n_layers=N_LAYERS, dropout_prob=DROPOUT):
        """
        Class constructor.
        :param input_output_dim: vocabulary size
        :param embedding_dim: embedding size
        :param hidden_dim: hidden state size
        :param n_layers: number of layers
        :param dropout_prob: dropout probability
        """

        super().__init__()

        self.input_output_dim = input_output_dim

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, input_output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, target_token,  hidden, cell):
        """
        Forward pass
        :param target_token: ground truth target sentence.
        :param hidden: hidden state
        :param cell: cell state
        :return: output, hidden state, and cell state
        """

        # Batch the target token
        target_token = target_token.unsqueeze(0)

        # Embedding layer
        embedding_layer = self.dropout(self.embedding(target_token))

        # LSTM cell
        output, (hidden, cell) = self.lstm(embedding_layer, (hidden, cell))

        # Fully connected layer
        linear = self.fc(output.squeeze(0))

        return linear, hidden, cell


class OneStepDecoderWithAttention(nn.Module):
    """
    The OneStepDecoder model with attention mechanism.
    """

    def __init__(self, input_output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention, dropout_prob=DROPOUT):
        """
        Class constructor.
        :param input_output_dim: vocabulary size
        :param embedding_dim: embedding size
        :param hidden_dim: hidden state size
        :param n_layers: number of layers
        :param dropout_prob: dropout probability
        """

        super().__init__()

        self.input_output_dim = input_output_dim
        self.attention = attention
        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.gru = nn.GRU(encoder_hidden_dim+embedding_dim, decoder_hidden_dim)
        self.fc = nn.Linear(encoder_hidden_dim+decoder_hidden_dim + embedding_dim, input_output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, target_token,  hidden, encoder_outputs):
        """
        Forward pass
        :param encoder_outputs:
        :param target_token: ground truth target sentence.
        :param hidden: hidden state
        :param cell: cell state
        :return: output, hidden state, and cell state
        """

        # Batch the target token
        target_token = target_token.unsqueeze(0)

        # Embedding layer
        embedding_layer = self.dropout(self.embedding(target_token))

        # Attention weights
        attention_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)

        # Fully connected layer
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        W = torch.bmm(attention_weights, encoder_outputs)

        W = W.permute(1, 0, 2)

        gru_input = torch.cat((embedding_layer, W), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        predicted_token = self.fc(torch.cat((output.squeeze(0), W.squeeze(0), embedding_layer.squeeze(0)), dim=1))

        return predicted_token, hidden, attention_weights.squeeze(1)


class Decoder(nn.Module):
    """
    The Decoder class
    """
    def __init__(self, one_step_decoder, device):
        """
        The class constructor.
        :param one_step_decoder: OneStepDecoder
        :param device: default device, 'cpu' or 'gpu'.
        """

        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, hidden, cell, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        """
        Forward pass.
        :param target: batch
            Target batch of sentences
        :param hidden: torch.tensor
            Hidden state of last time step, encoder's last hidden state as the first
            hidden state of the OneStepEncoder model.
        :param cell: Cell state of the time step, encoder's last hidden state as the first
        cell state of the OneStepEncoder model.
        :param teacher_forcing_ratio: double
            teacher forcing ratio, RNN networks training strategy.
            see also : https://arxiv.org/abs/1610.09038
        :return:
        """

        # print('target====================================', target)
        target_len, batch_size = target.shape[0], target.shape[1]

        # get the target language's vocabulary size
        target_vocab_size = self.one_step_decoder.input_output_dim

        # predicted indices
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        # the sos token will be the first input to the OneStepDecoder,
        # hence to LSTM cell at the first time step

        input_sequence = target[0, :]

        # go through all time step
        for time_step in range(1, target_len):

            # get output, hidden and cell state of the next time step
            predicted_index, hidden, cell = self.one_step_decoder(input_sequence, hidden, cell)

            # store the predicted indices at that time step
            predictions[time_step] = predicted_index

            # take the index of the maximum value
            predicted_sequence = predicted_index.argmax(1)

            # Apply teacher forcing technique
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                input_sequence = target[time_step]
            else:
                input_sequence = predicted_sequence

        return predictions


class DecoderWithAttention(nn.Module):
    """
    The Decoder class with attention
    """
    def __init__(self, one_step_decoder, device):
        """
        The class constructor.
        :param one_step_decoder: OneStepDecoder
        :param device: default device, 'cpu' or 'gpu'.
        """

        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, encoder_outputs, hidden, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        """
        Forward pass.

        :param target: batch
            Target batch of sentences
        :param encoder_outputs: Tensor
            Encoder outputs
        :param hidden: torch.tensor
            Hidden state of last time step, encoder's last hidden state as the first
            hidden state of the OneStepEncoder model.
        :param cell: Cell state of the time step, encoder's last hidden state as the first
        cell state of the OneStepEncoder model.
        :param teacher_forcing_ratio: double
            teacher forcing ratio, RNN networks training strategy.
            see also : https://arxiv.org/abs/1610.09038
        :return:
        """

        # print('target====================================', target)
        target_len, batch_size = target.shape[0], target.shape[1]

        # get the target language's vocabulary size
        target_vocab_size = self.one_step_decoder.input_output_dim

        # predicted indices
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        # the sos token will be the first input to the OneStepDecoder,
        # hence to LSTM cell at the first time step

        input_sequence = target[0, :]

        # go through all time step
        for time_step in range(1, target_len):

            # get output, hidden and cell state of the next time step
            predicted_index, hidden, cell = self.one_step_decoder(input_sequence, hidden, encoder_outputs)

            # store the predicted indices at that time step
            predictions[time_step] = predicted_index

            # take the index of the maximum value
            predicted_sequence = predicted_index.argmax(1)

            # Apply teacher forcing technique
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                input_sequence = target[time_step]
            else:
                input_sequence = predicted_sequence

        return predictions

