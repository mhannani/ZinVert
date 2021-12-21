from src.models.Encoder import Encoder
from src.models.Decoder import OneStepDecoder
from src.data.get_data import get_data
from src.data.Vocabulary import Vocabulary
from src.data.config import EMBEDDING_SIZE, HIDDEN_DIM, DROPOUT, N_LAYERS


if __name__ == "__main__":
    print('decoder model tests...')

    # get data
    train = get_data(batch_size=512, split='train')
    vocab = Vocabulary().build_vocab()
    vocab__de__len = len(vocab['de'])
    vocab__en__len = len(vocab['en'])
    print('vocab__de__len: ', vocab__de__len)

    # # feed the sos token as the first input to our one_step_encoder which contain LSTM cell
    one_step_decoder = OneStepDecoder(vocab__de__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)
    one_step_decoder.train()
    print('input_output_dim: ', one_step_decoder.input_output_dim)

    # encode source sentence.
    encoder = Encoder(vocab__en__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)
    encoder.train()
    for i, (src, tgt) in enumerate(train):
        encoder_hidden, encoder_cell = encoder(src)
        predict, decoder_hidden, decoder_cell = one_step_decoder(tgt[0, :], encoder_cell, encoder_cell)

        print('predict', predict)
        print('predict', predict.shape)
        print('decoder hidden size', decoder_hidden.shape)
        print('decoder cell size', decoder_cell.shape)
        break


