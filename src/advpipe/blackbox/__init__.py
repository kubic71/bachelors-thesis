
class TargetBlackBox:
    """
    Generic BlackBox target model
    Abstraction for cloud MLaaS classifiers as well as for local pretrained CNNs
    """

    blackbox_config : None

    def __init__(self, blackbox_config):
        self.blackbox_config = blackbox_config

    @staticmethod
    def fromConfig(blackbox_config):
        target_blackbox_list[blackbox_config.name](blackbox_config)
        

    def loss(self, pertubed_image):
        # Override this in Cloud/Local blackbox child class

        # In case of Cloud blackbox, query the API and pass it through the language-model translation mapping layer
        return 1
        

    

from .local import LocalBlackBox
target_blackbox_list = {"resnet18":}
