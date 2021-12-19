from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from Multi30k import CustomMulti30k
from config import *


class Vocabulary:
    """
    Build vocabulary for a language.
    """

    def __init__(self, dataset, freq_threshold):
        """
        The class constructor.
        :param dataset: dataset
        :param freq_threshold: int
            The frequency threshold for the vocabulary.
        """

        self.freq_threshold = freq_threshold
        self.dataset = dataset

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
            print(self._get_tokenizer(lang)(data_sample[LANGUAGE_INDEX[lang]]))
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
            vocabulary = build_vocab_from_iterator(self._get_tokens(data_iterator, lang), min_freq=self.freq_threshold,
                                                   specials=SPECIAL_SYMBOLS, special_first=True)
            vocabulary.set_default_index(UNK_IDX)

            return vocabulary
        else:
            raise ValueError('Not a supported language')

    def __call__(self):
        """
        Call the function when instantiation.
        :return: Set
            Set of the vocabulary of the two languages.
        """

        lang_vocabulary = {}
        for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
            print(language)
            lang_vocabulary[language] = self.build_vocab(self.dataset, lang=language)

        return lang_vocabulary


if __name__ == "__main__":
    custom_multi30k = CustomMulti30k(split=('train', 'valid', 'test'))
    train, valid, test = custom_multi30k.extract_sets()
    print(len(train))
    vocab = Vocabulary(train, 1)
    en_vo = vocab()['en'].get_stoi()
    de_vo = vocab()['en'].get_stoi()

    print(en_vo)

