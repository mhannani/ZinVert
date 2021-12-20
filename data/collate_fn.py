from config import *
from Vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence


class CollateFn:
    """
    Class to collate sample into batch with the same length (sentence length).
    """

    def __init__(self):
        """
        Class constructor.
        """
        self.pad_index = PAD_IDX
        self.vocab = Vocabulary(freq_threshold=1)
        self.text_transform = self.vocab.preprocess()

    def __call__(self, batch):
        """
        Allow the class to be called as function.
        :return:
        """

        # split the batch
        src_batch, tgt_batch = [], []

        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.pad_index)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_index)

        return src_batch, tgt_batch
