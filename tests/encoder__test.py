from src.models.Encoder import Encoder
from src.data.Vocabulary import Vocabulary
from src.data.get_data import get_data


if __name__ == "__main__":
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5
    epochs = 10

    vocab = Vocabulary()
    vocabulary = vocab.build_vocab()
    train_set = get_data(batch_size=512, split='train')
    vocab_size = len(vocabulary['en'])

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    encoder.train()

    for i, (src, tgt) in enumerate(train_set):
        hidden, cell = encoder(src)
        print('hidden', hidden)
        print('cell', cell)

        print('hidden size', hidden.shape)
        print('cell size', cell.shape)
        break
