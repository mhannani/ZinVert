def input_fn(request_body, request_content_type):
    """
    An input_fn that loads the pickled tensor by the inference server of SageMaker.
    The function deserialize the inference request, then the predict_fn get invoked.
    Does preprocessing and returns a tensor representation of the source sentence
    ready to give to the model to make inference.

    :param request_body: str
        The request body
    :param request_content_type: type
        The request body type.
    :return:
    """

