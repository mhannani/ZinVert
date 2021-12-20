
def train_one_epoch(model, optimizer):
    """
    Train model on single epoch with the optimizer.
    :param model: nn.Module
    :param optimizer: nn.Optimizer
    :return: double
        training loss during that epoch.
    """

    pass


def train(train_iter, valid_iter, src_vocab, tgt_vocab, epochs=10):
    """
        Train the seq2seq network for neural translation task.
    :param train_iter: Train set iterator
    :param valid_iter: Test set iterator
    :param src_vocab: Source language vocabulary.
    :param tgt_vocab: Source language vocabulary.
    :param epochs: number of epochs
    :return: Trained model.
    """

    pass


if __name__ == "__main__":
    print('preparation for training')

