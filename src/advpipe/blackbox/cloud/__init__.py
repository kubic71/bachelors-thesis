from __future__ import annotations
from typing import NewType, Sequence, Tuple
from advpipe.blackbox import TargetBlackBox
from advpipe.blackbox.loss import margin_loss
from advpipe.language_model.label_classification import OrganismLabelClassifier
from advpipe.log import logger, CloudDataLogger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import CloudBlackBoxConfig

CloudLabels = NewType('CloudLabels', Sequence[Tuple[str, float]])


class CloudBlackBox(TargetBlackBox):
    """Pretrained ImageNet model"""
    _organism_label_classifier = OrganismLabelClassifier()
    cloud_data_logger: CloudDataLogger

    def __init__(self, blackbox_config: CloudBlackBoxConfig):
        super().__init__(blackbox_config)
        self.cloud_data_logger = blackbox_config.cloud_data_logger

    def _loss(self, labels_and_scores: CloudLabels) -> float:
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

        for labels in [correct, adversarial]:
            if labels == []:
                labels.append(("no-label", 0))

        logger.debug(f"top-1 organism label: {correct[0][0]}, score: {correct[0][1]}")
        logger.debug(f"top-1 object label: {adversarial[0][0]}, score: {adversarial[0][1]}")

        if self.blackbox_config.loss.name == "margin_loss":
            loss_val = correct[0][1] - adversarial[0][1]
            return margin_loss(loss_val, self.blackbox_config.loss.margin)
        else:
            raise NotImplementedError(f"loss '{self.blackbox_config.loss.name}' not implemented!")


from .gvision_blackbox import GVisionBlackBox    # noqa: 402

CLOUD_BLACKBOXES = {"gvision": GVisionBlackBox}
