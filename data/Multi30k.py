from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
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
    vocab = custom_multi30k.build_vocab(train)
    # print('vocab_size', vocab.__len__())
    # print('itos --> ', vocab.get_itos())
    # print('itos --> ', vocab.get_stoi())
    print('+++++++++++++++++++++++++++++++++')
    train_dataloader = DataLoader(train, batch_size=10)
    print(type(train_dataloader))
    print('train test')
    for i, batch in enumerate(train_dataloader):
        print('..', {i})
        print(batch)


    # train_iter, _, _ = Multi30k(root="../.data", split=('train', 'valid', 'test'), language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # train_dataloader = DataLoader(train, batch_size=2)
    #
    # for i, batch in enumerate(train_dataloader):
    #     print('..', {i})
    #     print(batch)
