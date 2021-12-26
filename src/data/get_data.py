from .Multi30k import CustomMulti30k
from .Vocabulary import Vocabulary
from .collate_fn import CollateFn
from torch.utils.data import DataLoader
from src.data.config import *


def get_data(root='../data/.data', batch_size=BATCH_SIZE, split='train'):
    """
    Get the Multi30k dataset and vocabulary with both languages.
    :param root: str
        Path to the data, if not already there, download it
    :param batch_size: int
        Batch size.
    :param split: split name, train, or valid sets
    :return: List
        train, valid sets and both vocabulary of two languages.
    """

    train, valid, test = CustomMulti30k(root=root).extract_sets()
    sets = {'train': train, 'valid': valid, 'test': test}
    if split in ('train', 'valid', 'test'):
        iterator = DataLoader(sets[split][:10], batch_size=batch_size, collate_fn=CollateFn())

    else:
        raise ValueError('Split name not found !')

    return iterator

