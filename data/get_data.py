import torch
import spacy


def get_data(batch_size=128):
    """
    Download datasets, and extract vocabulary, returns train/test sets.
    :param batch_size: integer.
        Batch size.
    :return: train, test and valid sets along with pytext's Fields for both languages.
    """

    # load the languages
    spacy_en = spacy.load('en_core_web_sm')
    spacy_ar = spacy.load('fr_core_news_sm')

    # define the tokenizer for both languages


    return spacy_en, spacy_ar


if __name__ == "__main__":
    nlp_en, nlp_fr = get_data()
    sentence_en = "This world makes you happy if you deserve."
    sentence_fr = "Deux personnes se promenadent dans la rue."
    doc_en = nlp_en(sentence_en)
    doc_fr = nlp_fr(sentence_fr)

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
