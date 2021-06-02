import nltk
import os
from os import path
import sys
import re
from advpipe.log import logger


print("Loading wordnet")
nltk.download("wordnet")
from nltk import wordnet as wn  # pylint: disable=no-name-in-module



from functools import lru_cache


class OrganismLabelClassifier:

    # I've played with a suitable set of living/non-living category labels for huggingface zero shot classifier
    # These bellow work quite well on the ImageNet categoriees, except maybe for vegetable, which should be covered by wordnet anyway
    # That said, they are by no means optimal :)
    # Transformer is quite computationall, so we want to keep the list of labels small

    # What didn't work:
    # ["biological", "non-biological" ] - insects are classified as 'non-biological'
    # ['animate', 'inanimate'] - 'animate' is probably too fancy word for the transformerr
    # ['dead', 'alive'] - a lot of things and objects are classified as 'alive'
    #
    # large category set, with explicit specification of all the living things worked quite well, but was slow...
    # but maybe it's a good idea for the future, especially when the results will be cached to persistent storage
    zeroshot_categories = [
        "animal",
        "species",  # Animate labels
        "object",
        "instrument"
    ]  # Inanimate labels
    organism_categories = ["animal", "species"]

    wn_organism_synset = wn.wordnet.synsets("organism")[0]

    _transformer_classifier = None

    @property
    def classifier(self):
        # Lazy-load the transformer, because it's HUGE (1.52 GB) and loads slowly
        if OrganismLabelClassifier._transformer_classifier is None:
            print("Loading hugginnface transformer model")
            from transformers import pipeline

            script_dir = path.dirname(path.abspath(__file__))
            model_path = path.join(script_dir, "transformer_pretrained") 
            if not path.exists(path.join(script_dir, "transformer_pretrained")):
                # Download the transformer model and save it locally
                os.mkdir(model_path)
                OrganismLabelClassifier._transformer_classifier = pipeline("zero-shot-classification")
                OrganismLabelClassifier._transformer_classifier.save_pretrained(model_path)
            else:
                # load the downloaded pretrained model
                OrganismLabelClassifier._transformer_classifier = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path)

        return OrganismLabelClassifier._transformer_classifier


    def __init__(self):
        pass

    @lru_cache(maxsize=None)
    def is_organism_synset(self, synset_id):
        if not re.fullmatch(r"n[0-9]{8}", synset_id):
            raise ValueError(
                f"Invalid synset_id {synset_id} format. Valid synset_id is for example 'n02096294'"
            )

        offset = synset_id[1:]
        synset = wn.wordnet.synset_from_pos_and_offset('n', int(offset))
        if self._can_be_organism([synset]) and self._can_be_object([synset]):
            raise Exception(f"cannot classify synset {synset_id}, can be an organism and but also an object")
        return self._can_be_organism([synset])

    def _can_be_organism(self, synsets):
        """At least one synset is an organism"""
        for synset in synsets:
            for lch in synset.lowest_common_hypernyms(self.wn_organism_synset):
                logger.debug(
                    f"synset: {synset.name()}, lowest common hypernym: {lch.name()}"
                )
                if lch.name() == self.wn_organism_synset.name():
                    return True
        return False

    def _can_be_object(self, synsets):
        """At least one synset is an object"""
        for synset in synsets:
            for lch in synset.lowest_common_hypernyms(self.wn_organism_synset):
                logger.debug(
                    f"synset: {synset.name()}, lowest common hypernym: {lch.name()}"
                )
                if lch.name() != self.wn_organism_synset.name():
                    return True
        return False

    @lru_cache(maxsize=None)
    def is_organism_label(
        self,
        label,
    ):
        """Classify label as animate / inanimate

        :returns: True when label refers to a living organism 
        """
        label_synsets = wn.wordnet.synsets(label)

        logger.debug(f"is_organism executed with label {label}")

        # When wordnet fails, use transformer language model as a backup
        if label_synsets == []:
            logger.debug(
                f"label {label} not found in wordnet, falling back on transformer zero-shot"
            )
            top_label = self._transformer_classify(label)
            return top_label in self.organism_categories

        # Label was fond in WordNet
        possible_organism = self._can_be_organism(label_synsets)
        possible_object = self._can_be_object(label_synsets)

        if possible_organism and possible_object:
            # label has multiple meanings, let the transformer decide
            top_label = self._transformer_classify(label)
            return top_label in self.organism_categories
        else:
            return possible_organism

    def _transformer_classify(self, label):
        """Use huggingface zero-shot classifier for label classification"""

        result = self.classifier(label, self.zeroshot_categories) # pylint: disable=not-callable

        labels, scores = result["labels"], result["scores"]

        logger.debug(
            f"label {label} transformer classification:\nlabels:\t {labels}\nscores:\t{scores}"
        )

        top_label = labels[0]
        return top_label
