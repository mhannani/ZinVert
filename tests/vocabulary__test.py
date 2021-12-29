import torch
from src.data.Vocabulary import Vocabulary


def test_vocabulary_test():
    vocab = Vocabulary(freq_threshold=1)
    # vocab()  # calls the __call__ function (save vocabularies)

    en_voc, de_voc = vocab.vocabulary['en'], vocab.vocabulary['de']

    print(en_voc.get_stoi())
    print('==============================================')
    print(de_voc.get_stoi())

    print('vocabulary de size: ', len(de_voc.get_stoi()))
    print('vocabulary en size: ', len(en_voc.get_stoi()))

    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('The tensor_transform function testing')
    token_idx = [10, 20, 5, 100, 120, 302]
    print(vocab.tensor_transform(token_idx))  # working.

    print('++++++++++++++++++++++++++++++++++++++++++++++')

    print('preprocess sentences function test')
    phrase = "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt."
    print(vocab.preprocess()['de'](phrase))  # tensor([   2, 1167, 8948,  595,  132,    6,    3])
    print(vocab.postprocess(vocab.preprocess()['de'](phrase), 'de'))
