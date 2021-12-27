# Custom handler for torchserve deployment of the ZinVert model
import os
import json
import torch
import spacy
from ts.torch_handler.base_handler import BaseHandler


class ZinVertHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self):
        """
        The class constructor
        :return: None
        """

        super().__init__()
        self.itos_en_mappings = None
        self.stoi_en_mappings = None
        self.itos_de_mappings = None
        self.stoi_de_mappings = None

    def initialize(self, context):
        """

        :param context:
        :return:
        """

        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        itos_en_path = os.path.join(model_dir, "itos_en.json")
        stoi_en_path = os.path.join(model_dir, "stoi_en.json")
        itos_de_path = os.path.join(model_dir, "itos_de.json")
        stoi_de_path = os.path.join(model_dir, "stoi_de.json")

        with open(itos_en_path) as f:
            self.itos_en_mappings = json.load(f)

        with open(stoi_en_path) as f:
            self.stoi_en_mappings = json.load(f)

        with open(itos_de_path) as f:
            self.itos_de_mappings = json.load(f)

        with open(stoi_de_path) as f:
            self.stoi_de_mappings = json.load(f)

        # load the vocabularies
        self.initialized = True

    def preprocess(self, request):
        """

        :param request:
        :return:
        """
        request = request[0]
        request_body = request.get("body")
        dutch_sentence = request_body.get("dutch_sentence")
        tokens = self._get_tokens(dutch_sentence)

        # append <sos> and <eos> tokens
        tokens = ['<sos>'] + [token.lower() for token in tokens] + ['<eos>']

        sentence_indices = [self.stoi_de_mappings.get(token, 0) for token in tokens]

        return torch.LongTensor(sentence_indices).unsqueeze(1)

    @staticmethod
    def _get_tokens(sentence):
        """
        Get tokens of an  input sentence.
        :param sentence: str
            Input dutch sentence
        :return: list of tokens of the sentence.
        """

        # load dutch model
        dutch_tokenizer = spacy.load('de_core_news_sm')

        # get tokens
        doc = dutch_tokenizer(sentence)

        tokens = []

        for token in doc:
            tokens.append(token.text)

        return tokens

    def inference(self, data):
        """

        :param data:
        :return:
        """

        hidden, cell = self.model.encoder(data)

        # get the <sos> representation in vocabulary {As List} to be as the the first cell input
        sos_token_index = [2]

        sos_or_next_token_tensor = torch.LongTensor(sos_token_index)

        # output sentence
        trg_sentence = []

        # target indices predicted
        tgt_indices = []

        # disable gradient calculation
        with torch.no_grad():
            for _ in range(30):
                output, hidden, cell = self.model.decoder.one_step_decoder(sos_or_next_token_tensor, hidden, cell)
                sos_or_next_token_tensor = output.argmax(1)
                predicted_token = self.itos_en_mappings[sos_or_next_token_tensor.item()]
                # if we got <eos> then stop the loop
                if predicted_token == '<eos>':
                    break
                else:
                    # append the found index to target indices
                    tgt_indices.append(sos_or_next_token_tensor.item())
                    trg_sentence.append(predicted_token)

            predicted_words = [self.itos_en_mappings[i] for i in tgt_indices]

            return predicted_words

    def postprocess(self, data):
        """

        :param data:
        :return:
        """

        json_result = []
        en_sentence = " ".join(data)
        json_result.append({'caption': en_sentence})
        return json_result

    def handle(self, data, context):
        """

        :param data:
        :param context:
        :return:
        """

        sentence_tensor = self.preprocess(data)
        model_output = self.inference(sentence_tensor)
        en_sentence = self.postprocess(model_output)

        return en_sentence
