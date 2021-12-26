import torch
from termcolor import colored
from tqdm import tqdm
from time import time
from datetime import timedelta
from torch.nn.utils import clip_grad_norm_
from src.data.config import *
from src.data import get_data
from src.utils import create_seq2seq
from src.data import Vocabulary
from src.utils import save_model, load_checkpoints


def train(train_iter, valid_iter, src_vocab, tgt_vocab, epochs=EPOCHS, continue_training_checkpoints=None):
    """
    Train the seq2seq network for neural translation task.

    :param train_iter: Train set iterator
    :param valid_iter: Test set iterator
    :param src_vocab: Source language vocabulary(Dutch).
    :param tgt_vocab: Target language vocabulary(English).
    :param epochs: number of epochs
    :param continue_training_checkpoints: str
        Path to the checkpoint file to resume training from it
    :return: Trained model.
    """

    # create the model: model, optimizer, criterion
    seq2seq, optimizer, criterion = create_seq2seq(src_vocab, tgt_vocab)

    # starting epoch, will be 1 when training from scratch
    from_epoch = 1

    # Elapsed time during training, will be 0 seconds when training from scratch
    time_elapsed = 0
    # load the pretrained model with its learned weights
    if continue_training_checkpoints is not None:
        # load the checkpoint containing states of optimizer, model and the last epoch
        model_state_dict, optimizer_state_dict, from_epoch, time_elapsed, _, _ = load_checkpoints(continue_training_checkpoints)

        # update the state of the model with the saved state
        seq2seq.load_state_dict(model_state_dict)

        # update the state of the optimizer with the saved state
        optimizer.load_state_dict(optimizer_state_dict)

    # Training loop
    for epoch in range(from_epoch, epochs + 1):
        # compute the Elapsed time for the current training job
        start_time = time()

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

            # compute the loss
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

                # forward pass for validation data
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

        # Elapsed time of the current terminated job
        end_time = time()

        # Compute the total elapsed time of training
        # compute the elapsed time for that training job
        current_time_elapsed = end_time - start_time
        time_elapsed += current_time_elapsed

        # Save the checkpoint
        loss = round(sum(train_loss) / len(train_loss))
        save_model(seq2seq, optimizer, src_vocab, tgt_vocab, epoch, loss, time_elapsed)

        # save the JIT(Just In Time compilation) model

    print(colored('The training process of the model took: ', 'green'), colored(f'{timedelta(seconds=time_elapsed)}', 'red'))


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
    checkpoint = 'checkpoints/CHECKPOINT_WITHOUT_ATT__EN__TO__DE__EPOCH_3__AT__2021_12_25__23_36_38__TRAIN_LOSS__5.pt'
    train(train_iterator, valid_iterator, src_vocabulary, tgt_vocabulary, continue_training_checkpoints=None)
