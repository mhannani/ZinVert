from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from Multi30k import CustomMulti30k
import json
from config import *


class Vocabulary:
    """
    Build vocabulary for a language.
    """

    def __init__(self, freq_threshold):
        """
        The class constructor.
        :param dataset: dataset
        :param freq_threshold: int
            The frequency threshold for the vocabulary.
        """

        self.freq_threshold = freq_threshold
        self.vocabulary = self.build_vocab()

    def get_tokenizer(self):
        """
        Get the spacy tokenizer for the lang language.
        :param lang: str
            'en' for English or 'de' for Dutch.
        :return: spacy.tokenizer
        """

        token_transform = {}
        token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language=LANG_SHORTCUTS['en'])
        token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language=LANG_SHORTCUTS['de'])

        return token_transform

    def _get_tokens(self, data_iterator=None, lang='de'):
        """
        Get token for an iterator containing tuple of string
        :param lang: str
            'en' or 'de' for source and target languages.
        :return: List
            List of tokens.
        """
        tokenizer = self.get_tokenizer()
        for data_sample in data_iterator:
            # print(tokenizer[lang](data_sample[LANGUAGE_INDEX[lang]]))
            yield tokenizer[lang](data_sample[LANGUAGE_INDEX[lang]])

    def build_vocab(self):
        """
        Build the vocabulary of the given language.
        :param data_iterator: Iterator
            Data iterator.
        :return: List of Vocab
        """
        vocabulary = {}
        for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
            data_iterator = Multi30k(root="../.data", split='train', language_pair=('en', 'de'))
            vocabulary[lang] = build_vocab_from_iterator(self._get_tokens(data_iterator, lang), min_freq=self.freq_threshold,
                                                   specials=SPECIAL_SYMBOLS, special_first=True)
            vocabulary[lang].set_default_index(UNK_IDX)

        return vocabulary
    @staticmethod
    def _save_file(filename, data):
        # save the vocabulary as json
        with open(filename, 'w') as f:
            json.dump(data, f)

    def save_vocabulary(self, langs=('en', 'de')):
        """
        Save vocabulary to disk
        :return:
        """

        if 'en' not in langs and 'de' not in langs:
            raise ValueError('Not supported language(s) !')

        for language in langs:
            itos = self.vocabulary[language].get_itos()
            stoi = self.vocabulary[language].get_stoi()

            # save itos
            self._save_file(f'vocabulary/index_to_name_{language}', itos)

            # save stoi
            self._save_file(f'vocabulary/name_to_index_{language}', stoi)

    def __call__(self):
        """
        Call the function when instantiation.
        :return: Set
            Set of the vocabulary of the two languages.
        """

        self.save_vocabulary()


if __name__ == "__main__":
    vocab = Vocabulary(freq_threshold=1)
    vocab()

    en_voc, de_voc = vocab.vocabulary['en'], vocab.vocabulary['de']
    # print(en_voc.get_stoi())
    # en_vo_ = vo['en'].get_itos()
    # de_vo = vo['de'].get_stoi()
    # de_vo_ = vo['de'].get_itos()
    # print(en_vo)
    # print(en_vo_)
    #
    # print("========================")
    #
    # print(de_vo)
    # print(de_vo_)
    # vocab.save_vocabulary()

    # token_transform = vocab.get_tokenizer()
    # # token_transform = {}
    # # token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    # # token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
    #
    # vocab_transform = {}
    # # train_iter = Multi30k(root="../.data", split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #
    # # Define special symbols and indices
    # UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # # Make sure the tokens are in order of their indices to properly insert them in vocab
    # special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    #
    # for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    #     # Training data Iterator
    #     # print(ln)
    #     train_iter = Multi30k(root="../.data", split='train', language_pair=('en', 'de'))
    #     print(train_iter)
    #     # Create torchtext's Vocab object
    #     vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
    #                                                     min_freq=1,
    #                                                     specials=special_symbols,
    #                                                     special_first=True)
    #     print("=================================================================================")
    # print(vocab_transform['de'].get_stoi())


