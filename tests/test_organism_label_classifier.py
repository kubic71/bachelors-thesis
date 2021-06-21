# mypy: allow-untyped-defs, no-strict-optional
import pytest

from advpipe.language_model.label_classification import OrganismLabelClassifier


@pytest.fixture
def organism_label_classifier():
    return OrganismLabelClassifier()


organism_labels = [
    'Arctic Wolf',
    'Basilisk',
    'Black panther',
    'Bobolink',
    'Canid',
    'Cat',
    'Chicken',
    'Chimpanzee',
    'Ox',
    'Parrot',
    'Peregrine falcon',
    'Rook',
    'Siamese fighting fish',
    'Swordtail',
    'Turkey breeds',
    'Wildebeest',
    'Wolf'
    'Felidae',
    'Carnivore',
    'Small to medium-sized cats',
    'Terrestrial animal',
    'Plant',
    'Domestic short-haired cat',
    'Requiem Shark',
    'Lamniformes',
    'Great White Shark',
    'Carcharhiniformes',
    # 'Lamnidae', 'Crocodilia' # doesn't work, wordnet thinks it's an entity, not an organism
]

object_labels = [
    'money', 'monitor', 'camera', 'bow', 'button', 'lip gloss', 'green',
    'shawl', 'shirt', 'chocolate', 'phone', 'chalk', 'stop sign', 'canvas',
    'table', 'thermostat'
]

# Some ImageNet object labels
# Taken from here: https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
organism_synsets = [
    "n02119789",  # 1 kit_fox
    "n02100735",  # 2 English_setter
    "n02110185",  # 3 Siberian_husky
    "n02096294",  # 4 Australian_terrier
    "n02102040",  # 5 English_springer
    "n02066245",  # 6 grey_whale
    "n02509815",  # 7 lesser_panda
    "n02124075",  # 8 Egyptian_cat
    "n02417914",  # 9 ibex
    "n02123394",  # 10 Persian_cat
    "n02125311",  # 11 cougar
    "n02423022",  # 12 gazelle
    "n02346627",  # 13 porcupine
    "n02077923",  # 14 sea_lion
    "n02110063",  # 15 malamute
    "n02447366",  # 16 badger
    "n02109047",  # 17 Great_Dane
    "n02089867",  # 18 Walker_hound
    "n02102177",  # 19 Welsh_springer_spaniel
    "n02091134",  # 20 whippet
    "n02092002",  # 21 Scottish_deerhound
    "n02071294",  # 22 killer_whale
    "n02442845",  # 23 mink
    "n02504458",  # 24 African_elephant
    "n02092339",  # 25 Weimaraner
    "n02098105",  # 26 soft-coated_wheaten_terrier
    "n02096437",  # 27 Dandie_Dinmont
    "n02114712",  # 28 red_wolf
    "n02105641",  # 29 Old_English_sheepdog
    "n02128925",  # 30 jaguar
    "n02091635",  # 31 otterhound
    "n02088466",  # 32 bloodhound
    "n02096051",  # 33 Airedale
    "n02117135",  # 34 hyena
    "n02138441",  # 35 meerkat
    "n02097130",  # 36 giant_schnauzer
    "n02493509",  # 37 titi
    "n02457408",  # 38 three-toed_sloth
    "n02389026",  # 39 sorrel
    "n02443484",  # 40 black-footed_ferret
    "n02110341",  # 41 dalmatian
    "n02089078",  # 42 black-and-tan_coonhound
    "n02086910",  # 43 papillon
    "n02445715",  # 44 skunk
    "n09835506",  # 954 ballplayer - isn't an animal, but is an animate being (person)
    "n13037406",  # 956 gyromitra - fungi, which is an organism
]

# Some ImageNet object labels
object_synsets = [
    "n07920052",  # 947 espresso
    "n07873807",  # 948 pizza
    "n02895154",  # 949 breastplate
    "n04204238",  # 950 shopping_basket
    "n04597913",  # 951 wooden_spoon
    "n04131690",  # 952 saltshaker
    "n07836838",  # 953 chocolate_sauce
    "n03443371",  # 955 goblet
    "n04336792",  # 957 stretcher
    "n04557648",  # 958 water_bottle
    "n03187595",  # 959 dial_telephone
    "n04254120",  # 960 soap_dispenser
    "n03595614",  # 961 jersey
    "n04146614",  # 962 school_bus
    "n03598930",  # 963 jigsaw_puzzle
    "n03958227",  # 964 plastic_bag
    "n04069434",  # 965 reflex_camera
    "n03188531",  # 966 diaper
    "n02786058",  # 967 Band_Aid
    "n07615774",  # 968 ice_lolly
    "n04525038",  # 969 velvet
    "n04409515",  # 970 tennis_ball
    "n03424325",  # 971 gasmask
    "n03223299",  # 972 doormat
]


@pytest.mark.parametrize("organism_label", organism_labels)
def test_living_label(organism_label_classifier, organism_label):
    assert organism_label_classifier.is_organism_label(organism_label) is True


@pytest.mark.parametrize("object_label", object_labels)
def test_object_label(organism_label_classifier, object_label):
    assert organism_label_classifier.is_organism_label(object_label) is False


@pytest.mark.parametrize("synset_id", organism_synsets)
def test_living_synset_id(organism_label_classifier, synset_id):
    assert organism_label_classifier.is_organism_synset(synset_id) is True


@pytest.mark.parametrize("synset_id", object_synsets)
def test_object_synset_id(organism_label_classifier, synset_id):
    assert organism_label_classifier.is_organism_synset(synset_id) is False