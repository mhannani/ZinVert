class CollateFn:
    # src_batch, tgt_batch = [], []
    # for src_sample, tgt_sample in batch:
    #     src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
    #     tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    #
    # src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    # return src_batch, tgt_batch

    def __init__(self, pad_index):
        """
        Class constructor.
        """
        self.pad_index = pad_index

    def __call__(self, batch):
        """
        Allow the class to be called as function.
        :return:
        """

        # split the batch
        src_batch, target_batch = [], []


