from torchtext.datasets import Multi30k


class CustomMulti30k:
    """
    Custom class for Multi32K dataset.
    """
    def __init__(self, split, language_pair=('en', 'de')):
        """
        The class constructor.
        :param split: string
            'train', 'test', or 'valid' set to be returned.
        :param language_pair:
        """
        train_set, valid_set, test_set = Multi30k(split, language_pair)







if __name__ == "__main__":
    c = CustomMulti30k(split='train')






    
    