from src.models.Encoder import Encoder
from src.models.Decoder import OneStepDecoder, Decoder
from src.data.get_data import get_data
from src.data.Vocabulary import Vocabulary
from src.data.config import EMBEDDING_SIZE, HIDDEN_DIM, DROPOUT, N_LAYERS

if __name__ == "__main__":
    print('Decoder model tests...')

    # Get data
    train = get_data(batch_size=512, split='train')
    vocab = Vocabulary().build_vocab()
    vocab__de__len = len(vocab['de'])
    vocab__en__len = len(vocab['en'])

    encoder = Encoder(vocab__en__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)
    encoder.train()

    one_step_decoder = OneStepDecoder(vocab__de__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)
    one_step_decoder.train()

    decoder = Decoder(one_step_decoder, device='cpu')
    decoder.train()

    for i, (src, tgt) in enumerate(train):
        encoder_hidden, encoder_cell = encoder(src)
        predict = decoder(tgt, encoder_hidden, encoder_cell)
        print('predict', predict)
        print('predict', predict.shape)
        break
