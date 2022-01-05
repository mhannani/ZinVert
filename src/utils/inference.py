from src.data.config import *
from torch import LongTensor, no_grad
from src.utils.load_model import load_checkpoints
from src.utils.create_seq2seq import create_seq2seq
from src.utils.preprocess import preprocess
from src.utils.seq2seq_inference import seq2seq_inference
from torch import jit


def inference(sentence):
    """
    Doing inference with pretrained checkpoint.
    :param sentence: str
        A sentence
    :return: translated sentence
    """

    # Create model for inference
    seq_2_seq, src_vocabulary, tgt_vocabulary = seq2seq_inference(CHECKPOINT_PATH_WITHOUT_ATT)

    # loading just in time compilation
    # seq_2_seq = jit.load('checkpoints/JIT/model.pt')

    # print(seq_2_seq)
    # Preprocess the sentence
    sentence_tensor, tokens = preprocess(sentence, src_vocabulary)

    # Forward pass for the encoder
    hidden, cell = seq_2_seq.encoder(sentence_tensor)

    # get the <sos> representation in vocabulary {As List} to be as the the first cell input
    sos_token_index = [tgt_vocabulary.stoi['<sos>']]

    # convert it to tensor
    sos_or_next_token_tensor = LongTensor(sos_token_index)

    # output sentence
    trg_sentence = []

    # target indices predicted
    tgt_indices = []

    # disable gradient calculation
    with no_grad():
        # use the cell and hidden state from the Encoder model's farward pass
        # use these as the input hidden and cell state along side with
        # <sos> as the first token to the OneStepDecoder to generate token
        # in each time step of the decoder.

        # define the length of the sentence, aka number of cell in the model
        # when we reach the <sos> token, the generation process stops
        # and the sentence get returned.

        for _ in range(30):

            # get the output, hidden and cell state of the decoder

            output, hidden, cell = seq_2_seq.decoder.one_step_decoder(sos_or_next_token_tensor, hidden, cell)

            # the output length is the tgt_vocabulary length,
            # and it's a probability distribution of each token in target vocabulary
            sos_or_next_token_tensor = output.argmax(1)

            predicted_token = tgt_vocabulary.itos[sos_or_next_token_tensor.item()]

            # if we got <eos> then stop the loop
            if predicted_token == '<eos>':
                break
            else:
                # append the found index to target indices
                tgt_indices.append(sos_or_next_token_tensor.item())
                trg_sentence.append(predicted_token)

        predicted_words = [tgt_vocabulary.itos[i] for i in tgt_indices]

        return " ".join(predicted_words)
