from torchtext.datasets import Multi30k
from .config import *


class CustomMulti30k:
    """
    Custom class for Multi32K dataset.
    """

    def __init__(self, root, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
        """
        The class constructor.
        :param language_pair: default: ('de', 'en')
        """

        self.train, self.valid, self.test = Multi30k(root=root,
                                                     split=('train', 'valid', 'test'),
                                                     language_pair=language_pair)
    def extract_sets(self):
        """
        Extracts train, valid and test sets from Multi30K dataset.
        :return: List
        """

        return self.train, self.valid, self.test

