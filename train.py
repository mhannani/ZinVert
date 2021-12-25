import torch
from torch.nn.utils import clip_grad_norm_
from src.data.config import *
from src.data import get_data
from src.utils import create_seq2seq
from src.data import Vocabulary
from src.utils import save_model
from tqdm import tqdm
from time import gmtime, strftime


def train(train_iter, valid_iter, src_vocab, tgt_vocab, epochs=EPOCHS):
    """
    Train the seq2seq network for neural translation task.

    :param train_iter: Train set iterator
    :param valid_iter: Test set iterator
    :param src_vocab: Source language vocabulary(Dutch).
    :param tgt_vocab: Target language vocabulary(English).
    :param epochs: number of epochs
    :return: Trained model.
    """

    # create the model: model, optimizer, criterion
    seq2seq, optimizer, criterion = create_seq2seq(src_vocab, tgt_vocab)

    # Training loop
    for epoch in range(1, epochs + 1):
        # progress bar
        p_bar = tqdm(total=len(train_iter), bar_format='{l_bar}{bar:10}{r_bar}',
                     unit=' batches', ncols=200, mininterval=0.05, colour='#00ff00')

        train_loss = []

        # set train mode
        seq2seq.train()

        # loop through batches
        for src, tgt in train_iter:

            # see: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            # set gradients to zero
            optimizer.zero_grad()

            # forward pass
            outputs = seq2seq(src, tgt)

            # output dimension, corresponds to tgt_vocab__len
            output_dim = outputs.shape[-1]

            # discard first token
            output = outputs[1:].view(-1, output_dim)

            # discard <sos> token from target
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)

            # back propagation
            loss.backward()

            # clip gradient for stable network
            clip_grad_norm_(seq2seq.parameters(), 1)

            # update parameters
            optimizer.step()

            # save training loss during current batch pass
            train_loss.append(loss.item())

            # update the progress bar
            p_bar.set_postfix(epoch=f" {epoch}/{EPOCHS}, train loss= {round(sum(train_loss) / len(train_loss), 4)}, valid loss = +inf",
                             refresh=True)
            p_bar.update()

        # use trained model for the validation set
        with torch.no_grad():
            # turn eval mode
            seq2seq.eval()

            valid_loss = []
            for src, tgt in valid_iter:
                # forward pass
                outputs = seq2seq(src, tgt)

                # output dimension, corresponds to tgt_vocab__len
                output_dim = outputs.shape[-1]

                # discard first token
                output = outputs[1:].view(-1, output_dim)

                # discard <sos> token from target
                tgt = tgt[1:].view(-1)

                loss = criterion(output, tgt)

                # save validation loss during current batch pass
                valid_loss.append(loss.item())

        p_bar.set_postfix(
            epoch=f" {epoch}/{EPOCHS}, train loss= {round(sum(train_loss) / len(train_loss), 4)}, valid loss = {round(sum(train_loss) / len(train_loss), 4)}",
            refresh=False)
        p_bar.close()

        # Save the checkpoint
        if train_loss[-1] < train_loss[-2]:
            save_model(seq2seq, src_vocab, tgt_vocab, epoch=epoch, filename=f'checkpoints/CHECKPOINT_WITHOUT_ATT__EN__TO__DE__EPOCH_{epoch}__AT__{strftime("%Y_%m_%d__%H_%M_%S", gmtime())}__TRAIN_LOSS__{round(sum(train_loss) / len(train_loss))}')


if __name__ == "__main__":
    print('Training...')

    # Getting train DataLoader
    train_iterator = get_data(root='data/.data', batch_size=BATCH_SIZE, split='train')

    # Getting valid DataLoaders
    valid_iterator = get_data(root='data/.data', batch_size=BATCH_SIZE, split='valid')
    # valid_iterator = None

    # Initialize vocabulary
    vocab = Vocabulary()

    # Build vocabularies
    vocabularies = vocab.build_vocab()

    # Source and target vocabularies
    src_vocabulary = vocabularies['de']
    tgt_vocabulary = vocabularies['en']

    # Train network
    train(train_iterator, valid_iterator, src_vocabulary, tgt_vocabulary)

