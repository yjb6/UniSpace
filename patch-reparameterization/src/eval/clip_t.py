import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import SiglipModel, SiglipProcessor
from tqdm import tqdm
import os

# =================配置部分=================
# 本地模型路径 (包含 model.safetensors, config.json, preprocessor_config.json 等)
# MODEL_PATH = HF_HOME 下 siglip2-base-patch16-256
# MODEL_PATH = CKPT_ROOT 下 siglip2/test/ep-last/
MODEL_PATH = os.environ.get("SIGLIP2_MODEL_PATH", "google/siglip2-base-patch16-256")
# ImageNet 验证集根目录 (即包含 n01440764 等文件夹的上一级目录)
IMAGENET_VAL_PATH = os.path.join(os.environ.get("CLIP_BENCH_ROOT", "/path/to/clip_benchmark"), "imagenet-1k", "val")

BATCH_SIZE = 256
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================ImageNet 类别标签=================
# 标准 ImageNet-1k 的 1000 个类别名称 (对应 WordNet ID 排序)
# 为了代码简洁，这里列出前几个作为示例，完整列表请在代码下方查看说明或从外部加载
# 这里我们使用一个简单的函数来获取标签，通常实际评测会使用 OpenAI 的 1000 个类名
try:
    # 尝试从网络加载标准标签，如果无法联网请手动替换 IMAGENET_CLASSES 列表
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    IMAGENET_CLASSES = requests.get(url).text.strip().split('\n')
    print(f"成功加载 {len(IMAGENET_CLASSES)} 个类别标签。")
except Exception as e:
    print("无法下载标签，请确保联网或手动填入 IMAGENET_CLASSES。")
    # 示例: IMAGENET_CLASSES = ["tench", "goldfish", ...]
    # raise e
    IMAGENET_CLASSES = (
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

IMAGENET_CLASSES = tuple(label.lower() for label in IMAGENET_CLASSES)

# SigLIP 2 推荐的 Prompt 模板 (也可以使用 ensemble)
TEMPLATE = "a photo of a {}"

def load_model(model_path):
    """加载本地 SigLIP 模型和处理器"""
    print(f"正在加载模型: {model_path} ...")
    # SigLIP 2 在 HF 中通常复用 SiglipModel 结构
    model = SiglipModel.from_pretrained(model_path, torch_dtype=torch.float16)
    processor = SiglipProcessor.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, processor

def build_zero_shot_classifier(model, processor, class_names, template):
    """
    预计算文本嵌入 (Text Embeddings) 作为分类器权重
    """
    print("构建 Zero-Shot 分类器 (文本编码)...")

    # 构建 Prompts: ["a photo of a tench", "a photo of a goldfish", ...]
    texts = [template.format(label) for label in class_names]

    # 分批处理文本以防 OOM
    text_features_list = []
    batch_size = 256

    # 修复1: 使用 torch.amp.autocast('cuda') 替代过期的 torch.cuda.amp.autocast()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(
                text=batch_texts,
                padding="max_length",
                max_length=64, # 修复2: 显式指定 max_length (SigLIP 通常为 64)，解决 padding 失败导致的维度错误
                return_tensors="pt",
                truncation=True
            ).to(DEVICE)

            # 获取文本特征
            text_outputs = model.get_text_features(**inputs)

            # 归一化 (SigLIP/CLIP 必须步骤)
            text_outputs = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
            text_features_list.append(text_outputs)

    # 合并所有类别的特征 [1000, Hidden_Dim]
    text_classifier = torch.cat(text_features_list, dim=0)
    return text_classifier

class ImageNetDataset(torch.utils.data.Dataset):
    """
    自定义 Dataset 包装器，利用 SiglipProcessor 处理图像
    """
    def __init__(self, root_dir, processor):
        # 使用 ImageFolder 自动处理文件夹结构和标签索引
        self.dataset = datasets.ImageFolder(root_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 这里的 image 是 PIL Image
        return image, label

def collate_fn(batch, processor):
    """
    自定义 collate_fn，将 PIL 图片列表转换为 Tensor
    """
    images, labels = zip(*batch)
    # processor 处理图片: Resize, Rescale, Normalize
    # 修复: zip(*batch) 返回的是 tuple，必须转为 list 才能被 processor 接受
    inputs = processor(images=list(images), return_tensors="pt")
    return inputs["pixel_values"], torch.tensor(labels)

def evaluate(model, classifier, dataloader):
    """执行评测循环"""
    print("开始评测 ImageNet-1k...")
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    # SigLIP 的 Logit Scale (Temperature)
    logit_scale = model.logit_scale.exp()
    # 如果是 SigLIP 2，bias 也是需要的
    logit_bias = model.logit_bias if hasattr(model, 'logit_bias') else 0

    # 修复3: 同步更新 evaluate 中的 autocast 写法
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for pixel_values, targets in tqdm(dataloader):
            pixel_values = pixel_values.to(DEVICE).to(torch.float16)
            targets = targets.to(DEVICE)

            # 1. 图像编码
            # SigLIP image encoder 输出
            image_features = model.get_image_features(pixel_values=pixel_values)

            # 2. 归一化图像特征
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # 3. 计算 Logits
            # Logits = (Image_Feat @ Text_Feat.T) * Scale + Bias
            logits = (torch.matmul(image_features, classifier.t()) * logit_scale) + logit_bias

            # 4. 计算准确率
            # Top-1
            pred_top1 = logits.argmax(dim=1)
            top1_correct += (pred_top1 == targets).sum().item()

            # Top-5
            _, pred_top5 = logits.topk(5, dim=1)
            top5_correct += (pred_top5 == targets.unsqueeze(1)).sum().item()

            total_samples += targets.size(0)

    acc1 = top1_correct / total_samples
    acc5 = top5_correct / total_samples

    print(f"\n====== 评测完成 ======")
    print(f"Total Samples: {total_samples}")
    print(f"Top-1 Accuracy: {acc1 * 100:.2f}%")
    print(f"Top-5 Accuracy: {acc5 * 100:.2f}%")

    return acc1, acc5

# =================主执行逻辑=================
if __name__ == "__main__":
    from functools import partial

    # 1. 加载模型
    # 确保你的本地路径正确
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型路径: {MODEL_PATH}")

    model, processor = load_model(MODEL_PATH)

    # 2. 构建 Zero-Shot 文本分类器
    # 注意：IMAGENET_CLASSES 必须与 datasets.ImageFolder 的类别顺序一致
    # ImageFolder 默认按照文件夹名称(n01440764...)的字母顺序排序
    # 这里我们假设下载的 IMAGENET_CLASSES 是已经按照 WordNet ID 排序好的
    classifier = build_zero_shot_classifier(model, processor, IMAGENET_CLASSES, TEMPLATE)

    # 3. 准备数据加载器
    # 使用 partial 将 processor 传入 collate_fn
    collate_func = partial(collate_fn, processor=processor)

    dataset = ImageNetDataset(IMAGENET_VAL_PATH, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_func,
        pin_memory=True
    )

    # 4. 开始评测
    evaluate(model, classifier, dataloader)