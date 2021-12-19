from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from config import *


class Vocabulary:
    """
    Build vocabulary for a language.
    """

    def _get_tokenizer(self, lang):
        """
        Get the spacy tokenizer for the lang language.
        :param lang: str
            'en' for English or 'de' for Dutch.
        :return: spacy.tokenizer
        """

        if lang not in LANG_SHORTCUTS:
            raise ValueError('Language not found')
        else:
            return get_tokenizer('spacy', language=LANG_SHORTCUTS[lang])

    def _get_tokens(self, data_iterator=None, lang='en'):
        """
        Get token for an iterator containing tuple of string
        :param lang: str
            'en' or 'de' for source and target languages.
        :return: List
            List of tokens.
        """
        for data_sample in data_iterator:
            yield self._get_tokenizer(lang)(data_sample[LANGUAGE_INDEX[lang]])

    def build_vocab(self, data_iterator, lang='en'):
        """
        Build the vocabulary of the given language.
        :param data_iterator: Iterator
            Data iterator.
        :param lang: str
            Language shortcut.
        :return: None
        """

        # for English lang
        if lang in LANG_SHORTCUTS:
            vocabulary = build_vocab_from_iterator(self._get_tokens(data_iterator, lang), min_freq=1,
                                                   specials=SPECIAL_SYMBOLS, special_first=True)
            vocabulary.set_default_index(UNK_IDX)

            return vocabulary
        else:
            raise ValueError('Not a supported language')


if __name__ == "__main__":
    vocab = Vocabulary()

