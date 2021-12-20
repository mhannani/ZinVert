from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from collate_fn import CollateFn
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
        self.split = split

        self.train = Multi30k(root="../.data", split='train', language_pair=language_pair)
        self.valid = Multi30k(root="../.data", split='valid', language_pair=language_pair)
        self.test = Multi30k(root="../.data", split='test', language_pair=language_pair)

    def extract_sets(self):
        """
        Extracts train, valid and test sets from Multi30K dataset.
        :return: List
        """

        return self.train, self.valid, self.test


if __name__ == "__main__":
    custom_multi30k = CustomMulti30k(split=('train', 'valid', 'test'))
    train, valid, test = custom_multi30k.extract_sets()
    print('DATASET SUMMARY')
    print('+++++++++++++++')
    print(f'+ Train test: {train.__len__()} sentences')
    print(f'+ Valid test: {valid.__len__()} sentences')
    print(f'+ Test test: {test.__len__()} sentences')
    print('+++++++++++++++++++++++++++++++++++++++++++')
    train_dataloader = DataLoader(train, batch_size=10, collate_fn=CollateFn())
    print('train test')
    for i, (src, target) in enumerate(train_dataloader):
        print('..', {i})
        print('src: ', src)
        print('target: ', target)
        break
