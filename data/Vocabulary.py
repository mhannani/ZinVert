from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
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

    @staticmethod
    def get_tokenizer():
        """
        Get the spacy tokenizer for the lang language.
        :param lang: str
            'en' for English or 'de' for Dutch.
        :return: spacy.tokenizer
        """

        token_transform = {SRC_LANGUAGE: get_tokenizer('spacy', language=LANG_SHORTCUTS['en']),
                           TGT_LANGUAGE: get_tokenizer('spacy', language=LANG_SHORTCUTS['de'])}

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

    def save_vocabulary(self, lang=('en', 'de')):
        """
        Save vocabulary to disk
        :return:
        """

        if 'en' not in lang and 'de' not in lang:
            raise ValueError('Not supported language(s) !')

        for language in lang:
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
    vocab()  # calls the __call__ function (save vocabularies)

    en_voc, de_voc = vocab.vocabulary['en'], vocab.vocabulary['de']
    print(en_voc.get_stoi())
    print('==============================================')
    print(de_voc.get_stoi())
