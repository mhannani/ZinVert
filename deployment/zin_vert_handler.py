# Custom handler for torchserve deployment of the ZinVert model
from ts.torch_handler.base_handler import BaseHandler


class ZinVertHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init(self):
        """
        The class constructor
        :return: None
        """
