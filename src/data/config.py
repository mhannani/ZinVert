SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'de'
LANGUAGE_INDEX = {'en': 0, 'de': 1}
LANG_SHORTCUTS = {'en': 'en_core_web_sm', 'de': 'de_core_news_sm'}
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>']

EMBEDDING_SIZE = 256
HIDDEN_DIM = 1024
DROPOUT = 0.5
N_LAYERS = 2
EPOCHS = 25
BATCH_SIZE = 512
