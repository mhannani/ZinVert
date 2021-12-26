torch-model-archiver --model-name zin_vert_without_att \
--version 1.0 \
--serialized-fil ../checkpoints/CHECKPOINT_WITHOUT_ATT__EN__TO__DE__EPOCH_1__AT__2021_12_26__17_16_04__TRAIN_LOSS__5.pt \
--handler zin_vert_handler.py  \
--extra-files constants.pkl \
--export-path model-store -f