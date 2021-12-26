# Custom handler for torchserve deployment of the ZinVert model
import os
import torch
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

    def initialize(self, context):
        """
        Try to load torchscript else load the eager mode dict_state based model.
        :param context: The context
        :return: None
        """

        print('=======================INITIALIZE METHOD BODY============================== ')

        #  load the model
        self.manifest = context.manifest

        print('self.manifest: ', self.manifest)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        print('model_pt_path: ', model_pt_path)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        #
        print('model_pt_path:', model_pt_path)
        # self.model = torch.jit.load(model_pt_path)
        #
        # print(self.model)
        # self.initialized = True

    def preprocess(self, request):
        """

        :param request:
        :return:
        """

        # print(request)

    def inference(self, data):
        """

        :param data:
        :return:
        """

        pass

    def postprocess(self, data):
        """

        :param data:
        :return:
        """

        pass

    def handle(self, data, context):
        """

        :param data:
        :param context:
        :return:
        """

        return {}
