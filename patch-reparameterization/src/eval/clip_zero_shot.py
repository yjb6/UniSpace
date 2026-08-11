#!/usr/bin/env python3
"""
Zero-shot evaluation script for SigLIP2.

Modified from OpenCLIP.

Usage:
    python clip_zero_shot.py --model_path /path/to/siglip2_model --data_path /path/to/imagenet/val

    # With custom settings
    python clip_zero_shot.py \
        --model_path /path/to/siglip2_model \
        --data_path /path/to/imagenet/val \
        --batch_size 256 \
        --device cuda:0 \
        --precision bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import string
import re

from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel

logger = logging.getLogger(__name__)


# ============================================================================
# ImageNet classnames and templates (copied from tools/zero_shot/templates.py)
# ============================================================================

IMAGENET_CLASSNAMES = (
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
)

CLIP_PAPER_PROMPT_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.',
    'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
    'a photo of the cool {}.', 'a close-up photo of a {}.',
    'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
    'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
    'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.',
    'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.',
    'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
    'a photo of a small {}.', 'a tattoo of the {}.', '{}',
]
def canonicalize_text(text: str, *, keep_punctuation_exact_string: Optional[str] = None) -> str:
    """Canonicalize text (lowercase, remove punctuation, normalize whitespace).
    From big_vision.evaluators.proj.image_text.prompt_engineering.
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def get_canonicalized_class_names(class_names: List[str], first_only: bool = True) -> List[str]:
    """Canonicalize class names, optionally keeping first alias only."""
    out = []
    for name in class_names:
        n = canonicalize_text(name, keep_punctuation_exact_string=",")
        if first_only:
            n = n.split(",")[0].strip() if "," in n else n
        out.append(n)
    return out


def get_canonicalized_templates(templates: List[str]) -> List[str]:
    """Canonicalize prompt templates, preserving '{}' placeholder."""
    return [
        canonicalize_text(t, keep_punctuation_exact_string="{}")
        for t in templates
    ]
def substitute_prompt(template: str, class_name: str) -> str:
    """Substitute '{}' in template with class_name. From discriminative_classifier."""
    placeholder = "{}"
    parts = template.split(placeholder)
    assert len(parts) == 2, f"Template must have exactly one '{placeholder}': {template}"
    return parts[0] + class_name + parts[1]
# ============================================================================
# Utility functions
# ============================================================================

def batched(iterable, n: int):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """Calculate top-k accuracy."""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_input_dtype(precision: str) -> torch.dtype:
    """Get input dtype based on precision string."""
    if precision in ('bf16', 'bfloat16'):
        return torch.bfloat16
    elif precision in ('fp16', 'float16'):
        return torch.float16
    return torch.float32


def get_autocast_context(precision: str, device_type: str = 'cuda'):
    """Get autocast context manager based on precision."""
    from contextlib import nullcontext
    if precision in ('bf16', 'bfloat16'):
        return partial(torch.amp.autocast, device_type=device_type, dtype=torch.bfloat16)
    elif precision in ('fp16', 'float16'):
        return partial(torch.amp.autocast, device_type=device_type, dtype=torch.float16)
    return nullcontext


# ============================================================================
# Zero-shot classifier
# ============================================================================

@torch.no_grad()
def build_zero_shot_classifier(
    model: AutoModel,
    processor,
    classnames: Sequence[str],
    templates: Sequence[Callable],
    num_classes_per_batch: int = 10,
    device: torch.device = None,
    use_tqdm: bool = True,
    canonicalize: bool = True

) -> torch.Tensor:
    """Build zero-shot classifier weights.

    Args:
        model: SigLIP2 model instance.
        processor: SigLIP processor for text tokenization.
        classnames: Sequence of class names.
        templates: Sequence of template functions.
        num_classes_per_batch: Number of classes to process per batch.
        device: Device to use.
        use_tqdm: Whether to show progress bar.

    Returns:
        Zero-shot classifier weights, shape (embed_dim, num_classes).
    """
    if canonicalize:
        classnames = get_canonicalized_class_names(list(classnames))
        templates = get_canonicalized_templates(list(templates))

    num_templates = len(templates)
    num_classes = len(classnames)
    # 将classnames全部转为小写
    classnames = [c.lower() for c in classnames]
    if use_tqdm:
        num_iter = (num_classes - 1) // num_classes_per_batch + 1
        iter_wrap = partial(tqdm, total=num_iter, desc="Building classifier", unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames: List[str]) -> torch.Tensor:
        num_batch_classes = len(batch_classnames)
        texts = [
            substitute_prompt(tpl, c)
            for c in batch_classnames
            for tpl in templates
        ]
        # Use processor to tokenize texts
        inputs = processor(text=texts, return_tensors="pt", padding="max_length", max_length=64 , truncation=True)
        # inputs = processor(text=texts, return_tensors="pt", padding="max_length", max_length=64)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get text features from SigLIP model (with projection head)
        with torch.no_grad():
            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Normalize text features
        text_features = F.normalize(text_features, dim=-1)

        # Reshape and average over templates
        text_features = text_features.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        text_features = F.normalize(text_features, dim=1)
        return text_features.T  # (embed_dim, num_batch_classes)

    with torch.no_grad():
        batched_embeds = [
            _process_batch(batch)
            for batch in iter_wrap(batched(classnames, num_classes_per_batch))
        ]
        zeroshot_weights = torch.cat(batched_embeds, dim=1)

    return zeroshot_weights


# ============================================================================
# Evaluation
# ============================================================================
@torch.no_grad()
def evaluate(
    model: AutoModel,
    processor,
    classifier: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
    precision: str = 'fp32',
) -> Tuple[float, float]:
    """Run zero-shot evaluation.

    Args:
        model: SigLIP2 model instance.
        processor: SigLIP processor for image preprocessing.
        classifier: Zero-shot classifier weights, shape (embed_dim, num_classes).
        dataloader: ImageNet validation dataloader.
        device: Device to use.
        precision: Precision for inference.

    Returns:
        Tuple of (top1_accuracy, top5_accuracy).
    """
    autocast_ctx = get_autocast_context(precision, device_type=device.type)
    input_dtype = get_input_dtype(precision)

    top1, top5, n = 0.0, 0.0, 0

    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device=device, dtype=input_dtype)
            targets = targets.to(device)

            with autocast_ctx():
                # Get image features from SigLIP model (with projection head)
                image_features = model.get_image_features(pixel_values=images)

                # Normalize image features
                image_features = F.normalize(image_features, dim=-1)
                logits =  image_features @ classifier
                logits = model.logit_scale.exp() * logits + model.logit_bias
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n * 100
    top5 = top5 / n * 100
    return top1, top5


# ============================================================================
# PatchReparam encoder wrapper and dataloader for training integration
# ============================================================================


@torch.no_grad()
class PatchReparamZeroShotWrapper(nn.Module):
    """Wrapper to run zero-shot eval with PatchReparam encoder (image) + full SigLIP (text + logit).

    Used when integrating zero-shot into training: image features come from the
    current PatchReparam encoder (with output_head), text features and logit scale/bias
    from a frozen full SiglipModel.
    """

    def __init__(self, encoder: nn.Module, full_siglip: SiglipModel):
        super().__init__()
        self.encoder = encoder
        self._full_siglip = full_siglip
        # Expose logit parameters for evaluate()
        self.logit_scale = full_siglip.logit_scale
        self.logit_bias = full_siglip.logit_bias

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute image features using PatchReparam encoder + output head (same space as SigLIP)."""
        features = self.encoder.get_image_features(pixel_values)

        return F.normalize(features, dim=-1)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegate to full SigLIP text tower."""
        return self._full_siglip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


@torch.no_grad()
def create_imagenet_dataloader_patch_reparam(
    data_path: str,
    encoder_input_size: int,
    encoder_mean: Union[torch.Tensor, List[float]],
    encoder_std: Union[torch.Tensor, List[float]],
    batch_size: int = 128,
    num_workers: int = 8,
) -> DataLoader:
    """Create ImageNet val dataloader with PatchReparam encoder preprocessing (resize + normalize).

    Use this when evaluating PatchReparam encoder zero-shot so input matches encoder expectations.
    """
    if isinstance(encoder_mean, torch.Tensor):
        mean = encoder_mean.squeeze().tolist()
    else:
        mean = encoder_mean
    if isinstance(encoder_std, torch.Tensor):
        std = encoder_std.squeeze().tolist()
    else:
        std = encoder_std
    transform = transforms.Compose([
        transforms.Resize(encoder_input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(encoder_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = ImageFolder(root=data_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================================
# Main
# ============================================================================

def create_imagenet_dataloader(
    data_path: str,
    processor,
    batch_size: int = 128,
    num_workers: int = 8,
) -> DataLoader:
    """Create ImageNet validation dataloader using processor for preprocessing."""
    from PIL import Image

    class ImageNetDataset(ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = Image.open(path).convert('RGB')
            # Use processor to preprocess image
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dimension
            return pixel_values, target

    dataset = ImageNetDataset(root=data_path)
    # 验证类别顺序

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Zero-shot ImageNet evaluation for SigLIP2")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SigLIP2 model directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ImageNet validation dataset")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision for inference")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Zero-Shot ImageNet Evaluation (SigLIP2)")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load model and processor
    print("Loading model and processor...")
    # model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model = SiglipModel.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    # Debug: print model structure
    print(f"Model type: {type(model).__name__}")
    if hasattr(model, 'vision_model'):
        print(f"  Has vision_model: {type(model.vision_model).__name__}")
    if hasattr(model, 'text_model'):
        print(f"  Has text_model: {type(model.text_model).__name__}")
    if hasattr(model, 'get_image_features'):
        print(f"  Has get_image_features method")
    if hasattr(model, 'get_text_features'):
        print(f"  Has get_text_features method")
    print()

    # Get image size from processor
    if hasattr(processor, 'image_processor'):
        image_size = processor.image_processor.size
        print(f"Image size: {image_size}")
    print()

    # Create dataloader
    print(f"Loading ImageNet validation: {args.data_path}")
    dataloader = create_imagenet_dataloader(
        args.data_path,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataset = ImageFolder(root=args.data_path)
    class_to_idx = dataset.class_to_idx  # 字典：{文件夹名: 索引}
    sorted_classes = sorted(class_to_idx.items(), key=lambda x: x[1])  # 按索引排序

    # 检查前几个类别是否匹配
    for i, (folder_name, idx) in enumerate(sorted_classes[:10]):
        print(f"Index {idx}: {folder_name} -> {IMAGENET_CLASSNAMES[idx]}")

    num_samples = len(dataloader.dataset)
    print(f"Number of samples: {num_samples}")
    if num_samples != 50000:
        print(f"  [Warning] Expected 50000 samples for ImageNet validation, got {num_samples}")
    print()

    # Build classifier
    print("Building zero-shot classifier...")
    autocast_ctx = get_autocast_context(args.precision, device_type=device.type)
    with autocast_ctx():
        classifier = build_zero_shot_classifier(
            model,
            processor,
            classnames=IMAGENET_CLASSNAMES,
            templates=CLIP_PAPER_PROMPT_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
            canonicalize=True,
        )
    print()

    # Evaluate
    print("Running evaluation...")
    top1, top5 = evaluate(
        model,
        processor,
        classifier,
        dataloader,
        device=device,
        precision=args.precision,
    )

    print()
    print("=" * 60)
    print("Results:")
    print(f"  ImageNet Zero-Shot Top-1 Accuracy: {top1:.2f}%")
    print(f"  ImageNet Zero-Shot Top-5 Accuracy: {top5:.2f}%")
    print("=" * 60)

    results = {"top1": top1, "top5": top5}
    # Write results to model_path
    out_dir = Path(args.model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "zeroshot_imagenet_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top1": top1,
                "top5": top5,
                "data_path": args.data_path,
                "precision": args.precision,
                "batch_size": args.batch_size,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results written to {out_json}")

    return results


if __name__ == "__main__":
    main()
