# load model
import torch
from src.utils import load_checkpoints
from src.utils import create_seq2seq
from src.utils import preprocess

model_state_dict, optimizer_state_dict, epoch, time_elapsed, src_vocabulary, tgt_vocabulary = load_checkpoints('checkpoints/CHECKPOINT_WITHOUT_ATT__EN__TO__DE__EPOCH_16__AT__2021_12_27__10_16_46__TRAIN_LOSS__3.pt')
seq2seq, optimizer, criterion = create_seq2seq(src_vocabulary, tgt_vocabulary)

seq2seq.load_state_dict(model_state_dict)

# save jit mode model
de_sentence = 'Ein kleines Mädchen klettert in ein Spielhaus aus Holz.'
en_sentence = 'A little girl climbing into a wooden playhouse.'

seq2seq.eval()
# Trace the model

print('preprocess(de_sentence, src_vocabulary)[0]: ', preprocess(de_sentence, src_vocabulary)[0])
print('preprocess(de_sentence, src_vocabulary)[0]: ', preprocess(en_sentence, tgt_vocabulary)[0])
traced = torch.jit.trace(seq2seq,
                         (preprocess(de_sentence, src_vocabulary)[0], preprocess(en_sentence, tgt_vocabulary)[0]), check_trace=False)

# save traced model
traced.save('checkpoints/JIT/model.pt')