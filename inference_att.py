from src.utils.inference import inference_att


if __name__ == "__main__":
    print('Doing inference with attention based model...')

    # example sentence from test data
    de_sentence = "Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun."

    target_sentence = inference_att(de_sentence, is_jit=False)
    print(target_sentence)
