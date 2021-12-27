from src.utils.inference import inference


if __name__ == "__main__":
    print('Doing inference ...')

    # example sentence from test data
    de_sentence = 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'

    target_sentence = inference(de_sentence)
    print(target_sentence)
