import json
import torch


def save_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def save_mappings(checkpoint_name):
    """
    Load a pretrained checkpoint.
    :param checkpoint_name: checkpoint filename
    :return: model.state_dict, source_vocabulary, target_vocabulary,
    """

    # Get checkpoint from file
    checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))

    # Get source language vocabulary
    src_vocabulary = checkpoint['src_vocabulary']
    tgt_vocabulary = checkpoint['tgt_vocabulary']

    stoi_de_file = 'deployment/stoi_de.json'
    itos_de_file = 'deployment/itos_de.json'

    stoi_en_file = 'deployment/stoi_en.json'
    itos_en_file = 'deployment/itos_en.json'

    stoi_de = src_vocabulary.stoi
    itos_de = src_vocabulary.itos

    stoi_en = tgt_vocabulary.stoi
    itos_en = tgt_vocabulary.itos

    save_file(stoi_de, stoi_de_file)
    save_file(itos_de, itos_de_file)

    save_file(stoi_en, stoi_en_file)
    save_file(itos_en, itos_en_file)


if __name__ == "__main__":
    checkpoint_with_att = 'checkpoints/ATTENTION_CHECKPOINTS/CHECKPOINT_WITH_ATT__DE__TO__EN__EPOCH_6__AT__2021_12_30__19_32_58__TRAIN_LOSS__2.pt'
    checkpoint_without_att = 'checkpoints/'
    save_mappings(checkpoint_with_att)
