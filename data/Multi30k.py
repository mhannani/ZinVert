from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from config import *


class CustomMulti30k:
    """
    Custom class for Multi32K dataset.
    """

    def __init__(self, split, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
        """
        The class constructor.
        :param split: string
            'train', 'test', or 'valid' set to be returned.
        :param language_pair:
        """

        self.train, self.valid, self.test = Multi30k(root="../.data", split=split, language_pair=language_pair)

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

    def build_vocab(self, lang):
        """
        Build the vocabulary of the given language.
        :param lang: str
            Language shortcut.
        :return: None
        """
        # for English lang
        if lang == 'en':
            pass
        elif lang == 'de':
            pass
        else:
            raise ValueError('Not supported language')

    def _extract_sets(self):
        """
        Extracts train, valid and test sets
        :return: List
        """
        return self.train, self.valid, self.test


if __name__ == "__main__":
    custom_multi30k = CustomMulti30k(split=('train', 'valid', 'test'))
    train, valid, test = custom_multi30k._extract_sets()
    print('DATASET SUMMARY')
    print('+++++++++++++++')
    print(f'+ Train test: {train.__len__()} sentences')
    print(f'+ Valid test: {valid.__len__()} sentences')
    print(f'+ Test test: {test.__len__()} sentences')
    print('+++++++++++++++++++++++++++++++++++++++++++')
    custom_multi30k.get_tokenizer('de')
