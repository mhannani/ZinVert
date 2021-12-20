from .Multi30k import CustomMulti30k
from .Vocabulary import Vocabulary
from .collate_fn import CollateFn
from torch.utils.data import DataLoader


def get_data(batch_size, split='train'):
    """
    Get the Multi30k dataset and vocabulary with both languages.
    :param batch_size: int
        Batch size.
    :param split: split name, train, or valid sets
    :return: List
        train, valid sets and both vocabulary of two languages.
    """

    train, valid, test = CustomMulti30k().extract_sets()
    sets = {'train': train, 'valid': valid, 'test': test}
    if split in ('train', 'valid', 'test'):
        iterator = DataLoader(sets[split], batch_size=batch_size, collate_fn=CollateFn())

    else:
        raise ValueError('Split name not found !')

    return iterator

