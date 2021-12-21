import torch
from src.data.Vocabulary import Vocabulary


if __name__ == "__main__":
    vocab = Vocabulary(freq_threshold=1)
    vocab()  # calls the __call__ function (save vocabularies)

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
    phrase = "Two young, White males are outside near many bushes."
    print(vocab.preprocess()['en'](phrase))  # tensor([   2, 1167, 8948,  595,  132,    6,    3])
    print(vocab.postprocess(torch.tensor([   2,    5,  602,    8,   10,  507,  467,  270, 3436,  246,   21,   20,
         116,   17,   15,   11,    9,    4,    3]), 'en'))
