import torch
import spacy
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torchtext.legacy.data import Field, BucketIterator


class GetDataset(DataLoader):
    """
    Download datasets, and extract vocabulary.
    """
    super.__init__()

    # def __init__(self, batch_size=128):
    #     """
    #     :param batch_size: integer.
    #         Batch size.
    #         # :return: train, test and valid sets along with pytext's Fields for both languages.
    #     """
    #
    #     # load the languages
    #     self.spacy_en = spacy.load('en_core_web_sm')
    #     self.spacy_fr = spacy.load('fr_core_news_sm')

    def _tokenizer(self, sentence, lang='en'):
        """
        Tokenize the given sentence using the
        :param sentence: string
            A sentence
        :param lang: spacy language files, aka: spacy_en or spacy_fr
        :return: array_like of strings
            tokens of the given sentence
        """
        if lang == 'en':
            sp_lang = self.spacy_en
            return [token.text for token in sp_lang.tokenizer(sentence)]
        elif lang == 'fr':
            sp_lang = self.spacy_en
            return [token.text for token in sp_lang.tokenizer(sentence)]
        else:
            print("Sorry, Not a supported language...")

    def _extract_fields(self, tokenizer):
        """
        Creates pytext's Fields
        :param tokenizer: The language tokenizer
        :return: Fields of source and target language.
        """

        source = Field


if __name__ == "__main__":
    # GetDataset class instantiation
    get_dataset = GetDataset()
    sentence_en = "This world makes you happy if you deserve."
    sentence_fr = "Deux personnes se promenadent dans la rue."
    doc_en = get_dataset.spacy_en(sentence_en)
    doc_fr = get_dataset.spacy_fr(sentence_fr)

    # extract nouns
    en_nouns = []
    for noun_en in doc_en.noun_chunks:
        en_nouns.append(noun_en)

    fr_nouns = []
    for noun_fr in doc_fr.noun_chunks:
        fr_nouns.append(noun_fr)

    # extract verbs
    en_verbs = []
    for verb_en in doc_en:
        if verb_en.pos_ == "VERB":
            en_verbs.append(verb_en)
    fr_verbs = []
    for verb_fr in doc_fr:
        if verb_fr.pos_ == "VERB":
            fr_verbs.append(verb_fr)
    print("English sentence: ", sentence_en)
    print("French sentence:", sentence_fr)
    print("_______________________________________")

    print("English sentence nouns: ", en_nouns)
    print("French sentence nouns: ", fr_nouns)
    print("_______________________________________")

    print("English sentence verbs: ", en_verbs)
    print("French sentence verbs: ", fr_verbs)
    print("_______________________________________")


