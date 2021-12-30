import torch.nn as nn


class Seq2Seq(nn.Module):
    """
    Seq2seq model, combining encoder and decoder models.
    """

    def __init__(self, encoder, decoder):
        """
        The class constructor.
        :param encoder:  The Encoder model.
        :param decoder:  The Decocer model.
        """

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio = 0.5):
        """
        The forward pass
        :param src: torch.Tensor(BATCH_SIZE, 37 (LENGTH_OF_LONGEST_SENTENCE_IN_CORPUS))
            Source sentences [English sentences as batch]
        :param tgt: torch.Tensor(BATCH_SIZE, 46 (LENGTH_OF_LONGEST_SENTENCE_IN_CORPUS))
            Target sentences [Dutch sentences as batch]
        :param teacher_forcing_ratio:
            Teacher forcing ratio for applying the technique.
        :return: Torch.Tensor()
            Decoder output, Torch.Size([45 * 512, 19215])
        """

        # encode the source sentence
        hidden, cell = self.encoder(src)

        # decode it from the encoder's hidden and cell state.
        # print('***********************************************************************')
        # print(tgt)
        # print('***********************************************************************')
        outputs = self.decoder(tgt, hidden, cell, teacher_forcing_ratio)

        return outputs
