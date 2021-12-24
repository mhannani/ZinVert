from src.utils.inference import inference


if __name__ == "__main__":
    print('Doing inference ...')

    # example sentence from test data
    de_sentence = 'Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.'

    target_sentence = inference(de_sentence)
    print(target_sentence)




