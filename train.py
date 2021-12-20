# from timeit import default_timer as timer
from src.data.get_data import get_data
from tqdm import tqdm


def train_one_epoch(train_iter, valid_iter, model, optimizer, epoch):
    """
    Train model on single epoch with the optimizer.

    :param train_iter: train set iterator.
    :param valid_iter: valid set iterator.
    :param model: nn.Module
    :param optimizer: nn.Optimizer
    :return: double
        training loss during that epoch.
    """
    # progress bar
    pbar = tqdm(total=len(train_iter), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

    for i, batch in enumerate(train_iter):
        print(batch)

    pbar.set_postfix(
        epoch=f" {epoch}, train loss= {round(0.11, 4)}", refresh=True)
    pbar.update()


def train(train_iter, valid_iter, src_vocab, tgt_vocab, epochs=5):
    """
    Train the seq2seq network for neural translation task.

    :param train_iter: Train set iterator
    :param valid_iter: Test set iterator
    :param src_vocab: Source language vocabulary.
    :param tgt_vocab: Source language vocabulary.
    :param epochs: number of epochs
    :return: Trained model.
    """

    # create the model: model, optimizer, criterion = create_model(src_vocab, tgt_vocab)

    for epoch in range(1, epochs + 1):
        train_one_epoch([1, 2, 3], None, None, None, epoch)


if __name__ == "__main__":
    print('preparation for training')
    train_iterator = get_data(10, split='valid')
    print('got data')
    for (src, tgt) in train_iterator:
        print(src, tgt)
        break
    # train(train_iter=train_iterator, valid_iter=valid_iterator, src_vocab=None, tgt_vocab=None, epochs=5)

