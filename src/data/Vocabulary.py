import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
import json
from config import *


class Vocabulary:
    """
    Builds and save vocabulary for a language.
    """

    def __init__(self, freq_threshold=1):
        """
        The class constructor.
        :param dataset: dataset
        :param freq_threshold: int
            The frequency threshold for the processed.
        """

        self.freq_threshold = freq_threshold
        self.vocabulary = self.build_vocab()

    @staticmethod
    def get_tokenizer():
        """
        Get the spacy tokenizer for the lang language.
        :param lang: str
            'en' for English or 'de' for Dutch.
        :return: spacy.tokenizer
        """

        token_transform = {SRC_LANGUAGE: get_tokenizer('spacy', language=LANG_SHORTCUTS['en']),
                           TGT_LANGUAGE: get_tokenizer('spacy', language=LANG_SHORTCUTS['de'])}

        return token_transform

    def _get_tokens(self, data_iterator=None, lang='de'):
        """
        Get token for an iterator containing tuple of string
        :param lang: str
            'en' or 'de' for source and target languages.
        :return: List
            List of tokens.
        """
        tokenizer = self.get_tokenizer()
        for data_sample in data_iterator:
            yield tokenizer[lang](data_sample[LANGUAGE_INDEX[lang]])

    def build_vocab(self):
        """
        Build the processed of the given language.
        :return: List of Vocabs
        """
        vocabulary = {}
        for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
            data_iterator = Multi30k(split='train', language_pair=('en', 'de'))
            vocabulary[lang] = build_vocab_from_iterator(self._get_tokens(data_iterator, lang),
                                                         min_freq=self.freq_threshold, specials=SPECIAL_SYMBOLS,
                                                         special_first=True)
            vocabulary[lang].set_default_index(UNK_IDX)

        return vocabulary

    @staticmethod
    def tensor_transform(tokens_idx):
        """
        Builds the representation of numericalized sentence as Tensor.

        Input : A List, [12, 1, 6, 12, 200, 100] this a transformed sentence.
        (apply itos function to get the original text-based sentence).

        Output : The same input with EOS and SOS tensor concatenated
        respectively to the end and the beginning of the input tensor.

        :param tokens_idx: List
            A transformed sentence with indices of each token in it.

        :return: Tensor
            Sentence with SOS and EOS tokens added.
        """

        return torch.cat((
            torch.tensor([SOS_IDX]),
            torch.tensor(tokens_idx),
            torch.tensor([EOS_IDX])
        ))

    @staticmethod
    def pipeline(*transforms):
        """
        Make a pipeline of many transformation to the given input data.

        :param transforms: List
            List of transformation as arguments to the function
        :return: Function with transformation.
        """
        def shot(sentence):
            """
            Applies transformations as input.
            :param sentence:str
            :return: Tensor
                Input as Tensor
            """
            print(sentence)
            for transform in transforms:
                sentence = transform(sentence)
            return sentence

        return shot

    def postprocess(self, tensor, lang):
        """
        Postprocess a Tensor and get the corresponding text-based sentence from it.
        :return: str
            A sentence.
        """
        sentence = self.vocabulary[lang].lookup_tokens(tensor.tolist())
        return sentence


    def preprocess(self):
        """
        Tokenize, numericalize and turn into tensor a sentence.

        :return: Dict
            The transformation to be applied to a text-based sentence.
        """

        sentence_transform = {}

        for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
            sentence_transform[lang] = self.pipeline(
                self.get_tokenizer()[lang],
                self.build_vocab()[lang],
                self.tensor_transform
            )

        return sentence_transform

    @staticmethod
    def _save_file(filename, data):
        # save the processed as json
        with open(filename, 'w') as f:
            json.dump(data, f)

    def save_vocabulary(self, lang=('en', 'de')):
        """
        Save processed to disk
        :return:
        """

        if 'en' not in lang and 'de' not in lang:
            raise ValueError('Not supported language(s) !')

        for language in lang:
            itos = self.vocabulary[language].get_itos()
            stoi = self.vocabulary[language].get_stoi()

            # save itos
            self._save_file(f'../data/processed/index_to_name_{language}', itos)

            # save stoi
            self._save_file(f'../data/processed/name_to_index_{language}', stoi)

    def __call__(self):
        """
        Call the function when instantiation.
        :return: Set
            Set of the processed of the two languages.
        """

        self.save_vocabulary()


if __name__ == "__main__":
    vocab = Vocabulary(freq_threshold=1)
    vocab()  # calls the __call__ function (save vocabularies)

    en_voc, de_voc = vocab.vocabulary['en'], vocab.vocabulary['de']
    print(en_voc.get_stoi())
    print('==============================================')
    print(de_voc.get_stoi())

    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('The tensor_transform function testing')
    token_idx = [10, 20, 5, 100, 120, 302]
    print(vocab.tensor_transform(token_idx))  # working.

    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('preprocess sentences function test')
    phrase = "Two young, White males are outside near many bushes."
    print(vocab.preprocess()['en'](phrase))  # tensor([   2, 1167, 8948,  595,  132,    6,    3])
    print(vocab.postprocess(torch.tensor([2,20,26,16,1170,809,18,58,85,337,1340,6,3]), 'en'))
