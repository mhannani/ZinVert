from src.data.config import *
from src.models.Encoder import EncoderAttention
from src.data.Vocabulary import Vocabulary
from src.data.get_data import get_data


def encoder_att_test():

    vocab = Vocabulary()
    vocabulary = vocab.build_vocab()
    train_set = get_data(batch_size=BATCH_SIZE, split='train')
    vocab_size = len(vocabulary['de'])

    encoder = EncoderAttention(vocab_size, EMBEDDING_SIZE, HIDDEN_DIM, n_layers=N_LAYERS, dropout_prob=DROPOUT)
    encoder.train()

    for i, (src, _) in enumerate(train_set):
        hidden, cell = encoder(src)
        print('hidden', hidden.shape)
        print('cell', cell.shape)
        break
