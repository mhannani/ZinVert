torchserve --stop \
&&\
rm -r ./logs \
&&\
torch-model-archiver --model-name zin_vert_without_att \
--version 1.0 \
--serialized-fil ./JIT/model.pt \
--handler zin_vert_handler.py  \
--extra-files constants.pkl \
--export-path model-store -f\
&&\
torchserve --start --ncs --model-store model-store --models zin_vert_without_att.mar
