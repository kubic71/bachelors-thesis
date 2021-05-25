import nltk

print("Loading wordnet")
nltk.download("wordnet")
from nltk import wordnet as wn

print("Loading transformer  hugginnfacee pipeline")
from transformers import pipeline

from functools import lru_cache


class AnimalLabelClassifier:

    # I've played with a suitable set of animal/non-animal category labels for huggingface zero shot classifier
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
            "animal", "species",    # Animal labels
            "object", "instrument"] # Non-animal labels
    animal_categories = ["animal", "species"]


    # Should this be an 'organismm', or rather an 'animal'?
    wn_organism_synset = wn.wordnet.synsets("organism")[0]


    def __init__(self):
        self.classifier =  pipeline("zero-shot-classification")

    @lru_cache
    def is_animal_label(self, label):
        """Classify label as animal / not-an-animal

        :returns: True when label refers to an animal
        """
        label_synsets = wn.wordnet.synsets(label)

        # When wordnet fails, use transformer language model as a backup
        if label_synsets == []:
            return __transformer_classify(label)

        # Label was fond in WordNet
        label_syn = label_synsets[0]

        lch = label_syn.lowest_common_hypernyyms(self.wm_organism_synset)[0]

        # TODO: does lowest common hypernym always exsist? 
        return  lch.name() == "Synset('organism.n01)"




    def __transformer_classify(self, label):
        """Use huggingface zero-shot classifier for label classification"""

        result = self.classifier(label, self.zeroshot_categories)
        labels, scores = result["labels"], result["scores"]

        top_label = labels[0]
        return top_label in self.animal_categories







