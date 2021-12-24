from torch.nn.utils import clip_grad_norm_
from src.data.config import *
from src.data.get_data import get_data
from src.utils.create_seq2seq import create_seq2seq
from src.data.Vocabulary import Vocabulary
from src.utils.save_model import save_model
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

    for epoch in range(1, epochs + 1):

        p_bar = tqdm(total=len(train_iterator),
                     bar_format='{l_bar}{bar:10}{r_bar}',
                     unit=' batches', ncols=200)
        print(f'epoch {epoch}')
        training_loss = []

        # set train mode
        seq2seq.train()

        # loop through batches
        for i, (src, tgt) in enumerate(train_iter):
            print('train_iterator: ', train_iter)
            print(f'under batch loop , epoch_{epoch}')
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
            print('loss: ', loss)

            # back propagation
            loss.backward()
            print(f'backward loss{i}: ', loss)

            # clip gradient for stable network
            clip_grad_norm_(seq2seq.parameters(), 1)

            # update parameters
            optimizer.step()

            print('loss_item: ', loss.item())

            # save training loss during current batch pass
            training_loss.append(loss.item())
            print('training loss: ', training_loss)
            print('training loss length: ', len(training_loss))
            # update the progress bar
            p_bar.set_postfix(epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}",
                             refresh=True)
            p_bar.update()

        p_bar.set_postfix(
            epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}",
            refresh=False)
        p_bar.close()

        # Save the checkpoint
        # save_model(seq2seq, src_vocab, tgt_vocab, f'checkpoints/CHECKPOINT__EN__TO__DE__EPOCH_{epoch}__AT__{strftime("%Y_%m_%d__%H_%M_%S", gmtime())}__TRAIN_LOSS__{round(sum(training_loss) / len(training_loss))}')


if __name__ == "__main__":
    print('Training...')

    # Getting train, and valid DataLoaders
    train_iterator = get_data(root='data/.data', batch_size=BATCH_SIZE, split='test')

    # Valid_iterator = get_data(root='data/.data', batch_size=BATCH_SIZE, split='valid')
    valid_iterator = None

    # Initialize vocabulary
    vocab = Vocabulary()

    # Build vocabularies
    vocabularies = vocab.build_vocab()

    # Source and target vocabularies
    src_vocabulary = vocabularies['de']
    tgt_vocabulary = vocabularies['en']

    # Train network
    train(train_iterator, valid_iterator, src_vocabulary, tgt_vocabulary)
