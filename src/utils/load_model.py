import torch


def load_checkpoints(checkpoint_name):
    """
    Load a pretrained checkpoint.
    :param checkpoint_name: checkpoint filename
    :return: model.state_dict, source_vocabulary, target_vocabulary,
    """

    # Get checkpoint from file
    checkpoint = torch.load(checkpoint_name)

    # Get state_dict of the model
    model_state_dict = checkpoint['model_state_dict']

    # get source language vocabulary
    src_vocabulary = checkpoint['src_vocabulary']
    tgt_vocabulary = checkpoint['tgt_vocabulary']

    return model_state_dict, src_vocabulary, tgt_vocabulary
