from src.data.config import *
from src.models.Seq2seq import Seq2Seq
from src.models.Decoder import Decoder, OneStepDecoder
from src.models.Encoder import Encoder
from src.data.get_data import get_data
from src.data.Vocabulary import Vocabulary


if __name__ == "__main__":

    # Initialize vocabulary for both languages
    vocab = Vocabulary()

    # build vocabulary
    vocabulary = vocab.build_vocab()

    # getting training data
    train_set = get_data(batch_size=BATCH_SIZE, split='train')

    # vocabulary length
    vocab__en__len = len(vocabulary['en'])
    vocab__de__len = len(vocabulary['de'])

    # encoder model
    encoder = Encoder(vocab__en__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # one step decoder model
    one_step_decoder = OneStepDecoder(vocab__de__len, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # decoder model
    decoder = Decoder(one_step_decoder, device='cpu')

    # encoder -> decoder
    seq2seq = Seq2Seq(encoder, decoder)

    # turn in train mode
    seq2seq.train()

    for i, (src, tgt) in enumerate(train_set):
        print('src sentence :', src)
        print('tgt sentence: ', tgt)

        print('src sentence shape: ', src.shape)
        print('tgt sentence shape: ', tgt.shape)

        seq2seq_outputs = seq2seq(src, tgt)

        print('outputs: ', seq2seq_outputs)
        print('outputs shape: ', seq2seq_outputs.shape)

        # output formatting
        output_dim = seq2seq_outputs.shape[-1]
        tgt = tgt[1:].view(-1)
        outputs = seq2seq_outputs[1:].view(-1, output_dim)

        print('decoder output shape collapsed: ', outputs.shape)
        print('ground truth target shape collapsed: ', tgt.shape)

        # break the loop
        break
