from torch.nn.utils import clip_grad_norm_
from src.data.config import *
from src.data.get_data import get_data
from src.utils.create_seq2seq import create_seq2seq
from src.data.Vocabulary import Vocabulary
from tqdm import tqdm


def train(train_iter, valid_iter, src_vocab, tgt_vocab, epochs=EPOCHS):
    """
    Train the seq2seq network for neural translation task.

    :param train_iter: Train set iterator
    :param valid_iter: Test set iterator
    :param src_vocab: Source language vocabulary.
    :param tgt_vocab: Source language vocabulary.
    :param epochs: number of epochs
    :return: Trained model.
    """

    # create the model: model, optimizer, criterion
    model, optimizer, criterion = create_seq2seq(src_vocab, tgt_vocab)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(total=len(train_iterator),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    unit=' batches', ncols=200)
        training_loss = []

        # set train mode
        model.train()

        # loop through batches
        for i, (src, tgt) in enumerate(train_iter):
            # see: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            # set gradients to zero\
            optimizer.zero_grad()

            print('src.shape ===== : ', src.shape)
            print('trg.shape======: ', tgt.shape)

            # forward pass
            outputs = model(src, tgt)
            print('output.shape: ', outputs.shape)

            # output dimension, corresponds to tgt_vocab__len
            output_dim = outputs.shape[-1]

            # discard first token
            output = outputs[1:].view(-1, output_dim)

            # discard <sos> token from target
            tgt = tgt[1:].view(-1)
            print('shapes', output.shape, tgt.shape)

            loss = criterion(outputs, tgt)
            #
            # # back propagation
            loss.backward()
            #
            # # clip gradient for stable network
            clip_grad_norm_(model.parameters(), 1)
            #
            # # update parameters
            optimizer.step()
            #
            # # save training loss during current batch pass
            training_loss.append(loss.item())
            #
            # # update the progress bar
            pbar.set_postfix(epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}",
                             refresh=True)
            pbar.update()
            pbar.close()


if __name__ == "__main__":
    print('Training...')

    # getting train, and valid DataLoaders
    train_iterator = get_data(root='data/.data', batch_size=BATCH_SIZE, split='train')
    valid_iterator = get_data(root='data/.data', batch_size=BATCH_SIZE, split='valid')

    # Initialize vocabulary
    vocab = Vocabulary()

    # build vocabularies
    vocabularies = vocab.build_vocab()

    # source and target vocabularies
    src_vocabulary = vocabularies['en']
    tgt_vocabulary = vocabularies['de']

    train(train_iterator, valid_iterator, src_vocabulary, tgt_vocabulary)

