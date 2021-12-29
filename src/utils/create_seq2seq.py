import torch.nn as nn
from torch.optim import Adam
from src.models.Seq2seq import Seq2Seq
from src.models.Decoder import Decoder, OneStepDecoder, OneStepDecoderWithAttention
from src.models.Encoder import Encoder, EncoderAttention
from src.data.config import *


def create_seq2seq(src_vocab, tgt_vocab):
    """
    Creates encoder, decoder, defines optimizer, and loss function.
    :param src_vocab: torchtext.vocab.vocab.Vocab
        source language vocabulary
    :param tgt_vocab: torchtext.vocab.vocab.Vocab
        target language vocabulary
    :return: model, optimizer, criterion
    see : https://datascience.stackexchange.com/questions/10250/what-is-the-difference-between-objective-error-criterion-cost-loss-fun/10263
    """

    # vocabularies size
    src_vocab__len = len(src_vocab)
    tgt_vocab__len = len(tgt_vocab)

    # encoder model
    encoder = EncoderAttention(src_vocab__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # one step decoder model
    one_step_decoder = OneStepDecoder(tgt_vocab__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # decoder model
    decoder = Decoder(one_step_decoder, device='cpu')

    # encoder -> decoder
    seq2seq = Seq2Seq(encoder, decoder)

    # move the model to device
    seq2seq.to('cpu')

    # Adam optimizer
    optimizer = Adam(seq2seq.parameters())

    # ignore padding indices
    # TGT_PAD_IDX = tgt_vocab.lookup_indices([SPECIAL_SYMBOLS[PAD_IDX]])[0]
    TGT_PAD_IDX = 1

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

    return seq2seq, optimizer, criterion


def create_seq2seq_with_att(src_vocab, tgt_vocab):
    """
    Creates encoder, decoder, defines optimizer, and loss function with the attention mechanism
    :param src_vocab: torchtext.vocab.vocab.Vocab
        source language vocabulary
    :param tgt_vocab: torchtext.vocab.vocab.Vocab
        target language vocabulary
    :return: model, optimizer, criterion
    see : https://datascience.stackexchange.com/questions/10250/what-is-the-difference-between-objective-error-criterion-cost-loss-fun/10263
    """

    # vocabularies size
    src_vocab__len = len(src_vocab)
    tgt_vocab__len = len(tgt_vocab)

    # encoder model
    encoder = Encoder(src_vocab__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # one step decoder model
    one_step_decoder = OneStepDecoderWithAttention(tgt_vocab__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # decoder model
    decoder = Decoder(one_step_decoder, device='cpu')

    # encoder -> decoder
    seq2seq = Seq2Seq(encoder, decoder)

    # move the model to device
    seq2seq.to('cpu')

    # Adam optimizer
    optimizer = Adam(seq2seq.parameters())

    # ignore padding indices
    # TGT_PAD_IDX = tgt_vocab.lookup_indices([SPECIAL_SYMBOLS[PAD_IDX]])[0]
    TGT_PAD_IDX = 1

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

    return seq2seq, optimizer, criterion








