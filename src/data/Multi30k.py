from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from .collate_fn import CollateFn
from .config import *


class CustomMulti30k:
    """
    Custom class for Multi32K dataset.
    """

    def __init__(self, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)):
        """
        The class constructor.
        :param split: string
            'train', 'test', or 'valid' set to be returned.
        :param language_pair:
        """

        self.train = Multi30k(split='train', language_pair=language_pair)
        self.valid = Multi30k(split='valid', language_pair=language_pair)
        self.test = Multi30k(split='test', language_pair=language_pair)

    def extract_sets(self):
        """
        Extracts train, valid and test sets from Multi30K dataset.
        :return: List
        """

        return self.train, self.valid, self.test

