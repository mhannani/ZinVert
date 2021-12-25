import torch


def load_checkpoints(checkpoint_name):
    """
    Load a pretrained checkpoint.
    :param checkpoint_name: checkpoint filename
    :return: model.state_dict, source_vocabulary, target_vocabulary,
    """

    # Get checkpoint from file
    checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))

    # The epoch when training has been left
    epoch = checkpoint['epoch']

    # Get state_dict of the model
    model_state_dict = checkpoint['model_state_dict']

    # Get the state_dict of the optimizer
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    # Get source language vocabulary
    src_vocabulary = checkpoint['src_vocabulary']
    tgt_vocabulary = checkpoint['tgt_vocabulary']

    return model_state_dict, optimizer_state_dict, epoch, src_vocabulary, tgt_vocabulary
