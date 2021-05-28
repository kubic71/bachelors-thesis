import nltk
import re
import logging

print("Loading wordnet")
nltk.download("wordnet")
from nltk import wordnet as wn

print("Loading hugginnface transformer module")
from transformers import pipeline

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
    classifier = pipeline("zero-shot-classification")

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
        return self._can_be_organism([synset])

    def _can_be_organism(self, synsets):
        """At least one synset is an organism"""
        for synset in synsets:
            for lch in synset.lowest_common_hypernyms(self.wn_organism_synset):
                logging.debug(
                    f"synset: {synset.name()}, lowest common hypernym: {lch.name()}"
                )
                if lch.name() == self.wn_organism_synset.name():
                    return True
        return False

    def _can_be_object(self, synsets):
        """At least one synset is an object"""
        for synset in synsets:
            for lch in synset.lowest_common_hypernyms(self.wn_organism_synset):
                logging.debug(
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

        logging.debug(f"is_organism executed with label {label}")

        # When wordnet fails, use transformer language model as a backup
        if label_synsets == []:
            logging.debug(
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

        result = self.classifier(label, self.zeroshot_categories)
        labels, scores = result["labels"], result["scores"]

        logging.debug(
            f"label {label} transformer classification:\nlabels:\t {labels}\nscores:\t{scores}"
        )

        top_label = labels[0]
        return top_label