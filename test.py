from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import json

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'de'
LANGUAGE_INDEX = {'en': 0, 'de': 1}
LANG_SHORTCUTS = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']



# Place-holders
token_transform = {}
vocab_transform = {}

# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        print(token_transform[language](data_sample[language_index[language]]))
        yield token_transform[language](data_sample[language_index[language]])



if __name__ == "__main__":
    # custom_multi30k = CustomMulti30k(split='train')
    # train = custom_multi30k.extract_sets()
    # print(len(train))
    # # vocab = Vocabulary(train, 1)
    # en_vo = vocab()['en'].get_stoi()
    # en_vo_ = vocab()['en'].get_itos()
    # de_vo = vocab()['de'].get_stoi()
    # de_vo_ = vocab()['de'].get_itos()
    # print(en_vo)
    # print(en_vo_)
    #
    # print("========================")
    #
    # print(de_vo)
    # print(de_vo_)
    # vocab.save_vocabulary()

    # vocab_transform = {}
    # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    #     vocab_transform[ln] = build_vocab_from_iterator(_get_tokens(train_iter, ln),
    #                                                     min_freq=1,
    #                                                     specials=SPECIAL_SYMBOLS,
    #                                                     special_first=True)

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(root=".data", split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)