from src.data.config import *
from src.models.Encoder import Encoder
from src.models.Decoder import Decoder, OneStepDecoder
from src.models.Seq2seq import Seq2Seq
from src.utils.load_model import load_checkpoints


def seq2seq_inference(checkpoint_filename):
    """
    Prepare seq2seq model with pretrained checkpoint ready for inference.
    :param checkpoint_filename: str
        Checkpoints path
    :return: model, src_vocabulary, tgt_vocabulary
    """
    # load checkpoint

    model_state_dict, _, _, _, src_vocabulary, tgt_vocabulary = load_checkpoints(checkpoint_filename)
    # vocabularies size
    src_vocabulary_len = len(src_vocabulary)
    tgt_vocabulary_len = len(tgt_vocabulary)

    # Instantiate the models
    encoder = Encoder(src_vocabulary_len, EMBEDDING_SIZE, HIDDEN_DIM, n_layers=N_LAYERS, dropout_prob=DROPOUT)
    one_step_decoder = OneStepDecoder(tgt_vocabulary_len, EMBEDDING_SIZE, HIDDEN_DIM, n_layers=N_LAYERS, dropout_prob=DROPOUT)
    decoder = Decoder(one_step_decoder, 'cpu')

    # create encoder-decoder model
    seq2seq = Seq2Seq(encoder, decoder)

    # move model to cpu
    model = seq2seq.to('cpu')

    # copy learned parameters
    model.load_state_dict(model_state_dict)

    # turn model evaluation mode
    seq2seq.eval()

    return model, src_vocabulary, tgt_vocabulary
