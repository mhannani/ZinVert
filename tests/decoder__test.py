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
        print('source sentence: ', src)
        print('target sentence: ', tgt)
        print('=======================')
        print('source sentence shape: ', src.shape)
        print('target sentence shape: ', tgt.shape)

        encoder_hidden, encoder_cell = encoder(src)
        output = decoder(tgt, encoder_hidden, encoder_cell)
        print('output: ', output.shape)

        output_dim = output.shape[-1]
        tgt = tgt[1:].view(-1)
        output = output[1:].view(-1, output_dim)

        print('decoder output shape collapsed: ', output.shape)
        print('ground truth target shape collapsed: ', tgt.shape)
        break
