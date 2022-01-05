from src.utils.inference import inference


if __name__ == "__main__":
    print('Doing inference ...')

    # example sentence from test data
    de_sentence = "Ein Mann lächelt einen ausgestopften Löwen an."

    target_sentence = inference(de_sentence)
