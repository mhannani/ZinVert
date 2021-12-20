from Multi30k import CustomMulti30k
from Vocabulary import Vocabulary
from collate_fn import CollateFn
from torch.utils.data import DataLoader


def get_data(batch_size):
    """
    Get the Multi30k dataset and vocabulary with both languages.
    :param batch_size: int
        Batch size.
    :return: List
        train, valid sets and both vocabulary of two languages.
    """

    train, valid, _ = CustomMulti30k().extract_sets()
    train_iter = DataLoader(train, batch_size=batch_size, collate_fn=CollateFn())
    valid_iter = DataLoader(valid, batch_size=batch_size, collate_fn=CollateFn())

    return train_iter, valid_iter


if __name__ == "__main__":
    tr, te = get_data(2)
    for i in tr:
        print(i)
        print('===')
