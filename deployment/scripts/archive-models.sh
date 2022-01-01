echo "Archiving vanilla seq2seq model"
torch-model-archiver --model-name zin_vert_without_att --version 1.0 --serialized-fil ./JIT/WITHOUT_ATTENTION/model.pt --extra-files ./itos_de.json,./itos_en.json,./stoi_de.json,./stoi_en.json --handler zin_vert_handler.py --export-path model-store -f

echo "Archiving seq2seq model with attention mechanism"
torch-model-archiver --model-name zin_vert_with_att --version 1.0 --serialized-fil ./JIT/WITH_ATTENTION/model.pt --extra-files ./itos_de.json,./itos_en.json,./stoi_de.json,./stoi_en.json --handler zin_vert_handler.py --export-path model-store -f