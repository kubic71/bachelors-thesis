


class TargetBlackBox:
    """
    Generic BlackBox target model
    Abstraction for cloud MLaaS classifiers as well as for local pretrained CNNs
    """

    def __init__(self, blackbox_config):
        self.blackbox_config = blackbox_config

    def loss(self, pertubed_image):
        raise NotImplementedError
        # Override this in Cloud/Local blackbox child class

        # In case of Cloud blackbox, query the API and pass it through the language-model translation mapping layer
        # In case of Local blackbox, pass split the ImageNet categories using imagenet_utils.py


from .local import LocalBlackBox

TARGET_BLACKBOX_TYPES = {"local":LocalBlackBox,
    "cloud":None  # TODO
}


def margin_loss():
    raise NotImplementedError

loss_function = {"margin_loss": margin_loss}

