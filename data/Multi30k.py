from torchtext.datasets import Multi30k
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
