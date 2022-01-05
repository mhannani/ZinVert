from src.utils.seq2seq_inference import seq2seq_inference_with_attention
from src.utils.preprocess import preprocess
from src.data.config import *
from torch import LongTensor, no_grad, zeros


def inference_att(sentence):
    """
    Doing inference with seq2seq model with attention mechanism.
    :param sentence: str
    :return: translated sentence
    """

    # Create model for inference
    seq_2_seq, src_vocabulary, tgt_vocabulary = seq2seq_inference_with_attention(CHECKPOINT_PATH_WITH_ATT)

    # Preprocess the sentence
    sentence_tensor, tokens = preprocess(sentence, src_vocabulary)

    encoder_outputs, hidden = seq_2_seq.encoder(sentence_tensor)
    # get the <sos> representation in vocabulary {As List} to be as the the first cell input
    sos_token_index = [2]

    sos_or_next_token_tensor = LongTensor(sos_token_index)

    # output sentence
    trg_sentence = []

    # target indices predicted
    tgt_indices = []

    # disable gradient calculation
    with no_grad():
        for _ in range(30):
            output, hidden, attention = seq_2_seq.decoder.one_step_decoder(sos_or_next_token_tensor, hidden,
                                                                       encoder_outputs)
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

        return predicted_words
