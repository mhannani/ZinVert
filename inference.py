from src.utils.inference import inference


if __name__ == "__main__":
    print('Doing inference ...')

    # example sentence from test data
    de_sentence = 'Ein hell gekleideter Mann fotografiert eine Gruppe von Männern in dunklen Anzügen und mit Hüten, die um eine Frau in einem trägerlosen Kleid herum stehen.'

    target_sentence = inference(de_sentence)
    print(target_sentence)
