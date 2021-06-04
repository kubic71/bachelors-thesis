from advpipe.blackbox import TargetBlackBox
from advpipe.language_model.label_classification import OrganismLabelClassifier
from advpipe.log import logger
from advpipe.blackbox.loss import LOSS_FUNCTIONS

class CloudBlackBox(TargetBlackBox):
    """Pretrained ImageNet model"""
    _organism_label_classifier = OrganismLabelClassifier()

    def __init__(self, blackbox_config):
        super().__init__(blackbox_config)


    def _loss(self, labels_and_scores):
        correct = []
        adversarial = []
        for label, score in labels_and_scores:
            if self._organism_label_classifier.is_organism_label(label):
                correct.append((label, score))
            else:
                adversarial.append((label, score))
        
        # sort by score
        correct.sort(key=lambda x: x[1], reverse=True)
        adversarial.sort(key=lambda x: x[1], reverse=True)

        for l in [correct, adversarial]:
            if l == []:
                l.append(("no-label", 0))


        logger.debug(f"top-1 organism label: {correct[0][0]}, score: {correct[0][1]}")
        logger.debug(f"top-1 object label: {adversarial[0][0]}, score: {adversarial[0][1]}")

        if self.blackbox_config.loss.name == "margin_loss":
            loss_val = correct[0][1] - adversarial[0][1]
            LOSS_FUNCTIONS["margin_loss"](loss_val, self.blackbox_config.loss.margin)

        return 


        


from .gvision_blackbox import GVisionBlackBox
CLOUD_BLACKBOXES = {
    "gvision": GVisionBlackBox
}