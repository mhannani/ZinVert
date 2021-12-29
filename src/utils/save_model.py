import torch
from src.utils import preprocess
from time import gmtime, strftime


def save_model(model, optimizer, src_vocabulary, tgt_vocabulary, epoch, loss, time_elapsed, is_jit=False):
    """
    Save trained model, along with source and target languages vocabularies.

    :param model: The trained model
    :param optimizer: The optimizer used to train the model
    :param src_vocabulary: source language vocabulary.
    :param tgt_vocabulary: target language vocabulary.
    :param epoch: epoch at which the model will be saved.
    :param loss: double
        The loss of the model.
    :param is_jit: boolean
        Whether to save the JIT version of the model
    :param time_elapsed: int
        Number of seconds that the model took to train until that point
    :return: None
    """

    # define checkpoint dictionary
    checkpoint = {
        'epoch': epoch + 1,
        'loss': loss,
        'time_elapsed': time_elapsed,
        'src_vocabulary': src_vocabulary,
        'tgt_vocabulary': tgt_vocabulary,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    # Save the checkpoint
    filename = f'checkpoints/CHECKPOINT_WITHOUT_ATT__DE__TO__EN__EPOCH_{epoch}__AT__{strftime("%Y_%m_%d__%H_%M_%S", gmtime())}__TRAIN_LOSS__{loss}.pt'
    jit_filename = f'checkpoints/JIT/JIT__CHECKPOINT_WITHOUT_ATT__DE__TO__EN__EPOCH_{epoch}__AT__{strftime("%Y_%m_%d__%H_%M_%S", gmtime())}__TRAIN_LOSS__{loss}.pt'

    # save checkpoint
    torch.save(checkpoint, filename)

    # if the JIT model required to be saved as well
    if is_jit:
        # save jit mode model
        de_sentence = 'Ein kleines Mädchen klettert in ein Spielhaus aus Holz.'
        en_sentence = 'A little girl climbing into a wooden playhouse.'

        # Trace the model
        traced = torch.jit.trace(model, (preprocess(de_sentence, src_vocabulary)[0], preprocess(en_sentence, tgt_vocabulary)[0]), check_trace=False)

        # save traced model
        traced.save(jit_filename)
