from torch import save


def save_model(model, src_vocabulary, tgt_vocabulary, epoch, filename):
    """
    Save trained model, along with source and target languages vocabularies.
    :param model: trained model
    :param src_vocabulary: source language vocabulary.
    :param tgt_vocabulary: target language vocabulary.
    :param epoch: epoch at which the model will be saved.
    :param filename: filename of the checkpoint.
    :return: None
    """

    # define checkpoint dictionary
    checkpoint = {
        'epoch': epoch + 1,
        'src_vocabulary': src_vocabulary,
        'tgt_vocabulary': tgt_vocabulary,
        'model_state_dict': model.state_dict()
    }

    # save checkpoint
    save(checkpoint, filename)

