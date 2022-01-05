import os
import torch
import random
import torch.nn as nn
from torch.nn.functional import softmax

# constants
N_LAYERS = 1
DROPOUT = 0.5
EMBEDDING_SIZE = 256
HIDDEN_DIM = 1024
TEACHER_FORCING_RATIO = 0.5


class EncoderAttention(nn.Module):
    """
    The Encoder variant model for seq2seq with attention mechanism.
    """

    def __init__(self, vocab_length, embedding_dim, hidden_dim, n_layers=N_LAYERS, dropout_prob=DROPOUT):
        """
        The class constructor.

        :param vocab_length: The vocabulary length
        :param embedding_dim: embedding dimension
        :param hidden_dim: hidden size dimension
        :param n_layers: number of layer in the encoder
        :param dropout_prob: dropout probability for applying dropout feature to avoid overfitting.
        """

        super().__init__()

        self.vocab_length = vocab_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        # convert input tensor to nn.Embeddings with dimension supplied.
        # more info here: https://stackoverflow.com/questions/50747947/embedding-in-pytorch
        self.embedding = nn.Embedding(self.vocab_length, self.embedding_dim)

        # GRU cell
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.n_layers, dropout=dropout_prob)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, batch):
        """
        Train the Encoder network.
        :param batch: A batch
        :return: hidden state, and cell
        """

        # Apply dropout to the embeddings
        embed = self.dropout(self.embedding(batch))

        # Get the output from the LSTM (hidden state and cell state).
        outputs, hidden = self.gru(embed)

        return outputs, hidden


class Attention(nn.Module):
    """
    The attention model
    src: https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html
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


class Seq2Seq(nn.Module):
    """
    Seq2seq model, combining encoder and decoder models.
    """

    def __init__(self, encoder, decoder):
        """
        The class constructor.
        :param encoder:  The Encoder model.
        :param decoder:  The Decocer model.
        """

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio = 0.5):
        """
        The forward pass
        :param src: torch.Tensor(BATCH_SIZE, 37 (LENGTH_OF_LONGEST_SENTENCE_IN_CORPUS))
            Source sentences [English sentences as batch]
        :param tgt: torch.Tensor(BATCH_SIZE, 46 (LENGTH_OF_LONGEST_SENTENCE_IN_CORPUS))
            Target sentences [Dutch sentences as batch]
        :param teacher_forcing_ratio:
            Teacher forcing ratio for applying the technique.
        :return: Torch.Tensor()
            Decoder output, Torch.Size([45 * 512, 19215])
        """

        # encode the source sentence
        hidden, cell = self.encoder(src)

        outputs = self.decoder(tgt, hidden, cell, teacher_forcing_ratio)

        return outputs


def load_checkpoint(model_dir):
    """
    Loads checkpoints from file
    :param model_dir: str
        The path where the model is located
    :return: tuple
    """

    with open(os.path.join(model_dir, 'model.pt'), 'rb') as filename:
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))

    # Get state_dict of the model
    model_state_dict = checkpoint['model_state_dict']

    # Get source language vocabulary
    src_vocabulary = checkpoint['src_vocabulary']
    tgt_vocabulary = checkpoint['tgt_vocabulary']

    return model_state_dict, src_vocabulary, tgt_vocabulary


def load_model_for_inference(model_dir):
    """
    Prepare seq2seq model with pretrained checkpoint with attention ready for inference.
    :param checkpoint_filename: str
        Checkpoints path
    :return: model, src_vocabulary, tgt_vocabulary
    """
    # load checkpoint

    model_state_dict, src_vocabulary, tgt_vocabulary = load_checkpoint(model_dir)

    # vocabularies size
    src_vocabulary_len = len(src_vocabulary)
    tgt_vocabulary_len = len(tgt_vocabulary)

    # Instantiate the models
    encoder = EncoderAttention(src_vocabulary_len, EMBEDDING_SIZE, HIDDEN_DIM)

    # attention layer
    attention = Attention(HIDDEN_DIM, HIDDEN_DIM)

    # one step decoder
    one_step_decoder = OneStepDecoderWithAttention(tgt_vocabulary_len, EMBEDDING_SIZE, HIDDEN_DIM, HIDDEN_DIM, attention)
    decoder = DecoderWithAttention(one_step_decoder, 'cpu')

    # create encoder-decoder model
    seq2seq = Seq2Seq(encoder, decoder)

    # move model to cpu
    model = seq2seq.to('cpu')

    # copy learned parameters
    model.load_state_dict(model_state_dict)

    # turn model evaluation mode
    seq2seq.eval()

    return model, src_vocabulary, tgt_vocabulary


def model_fn(model_dir):
    """
    Loads the model in SageMaker.
    :param model_dir: src
    :return:
    """

    # load model for inference from checkpoints

    return load_model_for_inference(model_dir)


if __name__ == "__main__":
    print(model_fn('./'))