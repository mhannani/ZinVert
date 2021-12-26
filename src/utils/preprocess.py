import spacy
from torch import LongTensor


def _get_tokens(sentence):
    """
    Get tokens of an  input sentence.
    :param sentence: str
        Input dutch sentence
    :return: list of tokens of the sentence.
    """

    # load dutch model
    dutch_tokenizer = spacy.load('de_core_news_sm')

    # get tokens
    doc = dutch_tokenizer(sentence)

    tokens = []

    for token in doc:
        tokens.append(token.text)

    tokens.reverse()

    return tokens


def preprocess(sentence, src_vocabulary):
    """
    Preprocess the input sentence.
    :param sentence: str
        A sentence
    :param src_vocabulary: vocab
        Source language vocabulary.
    :return: torch.tensor
    """

    # tokenize the sentence
    tokens = _get_tokens(sentence)

    # append <sos> and <eos> tokens
    tokens = ['<sos>'] + [token.lower() for token in tokens] + ['<eos>']

    sentence_indices = [src_vocabulary.get_stoi()[token] for token in tokens]

    sentence_tensor = LongTensor(sentence_indices).unsqueeze(1)

    return sentence_tensor, tokens


if __name__ == "__main__":
    sentence_tokens = _get_tokens('Leute Reparieren das Dach eines Hauses.')
    print(sentence_tokens)

