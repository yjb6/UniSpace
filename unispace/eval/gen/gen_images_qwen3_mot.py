# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3 text-only MOT (Mixture-of-Transformers) 版本的图像生成测试脚本。

与 gen_images_mot.py 功能相同，但使用 Qwen3 text-only LLM（非 VL）：
  - Qwen3MoTForConditionalGeneration 替代 Qwen3VLMoTForConditionalGeneration
  - Qwen3Config 替代 Qwen3VLConfig
  - UnimmMoT 替代基于 Qwen3-VL 的 UnimmMoT
  - use_qwen_vit=False（不使用 Qwen3-VL 内置 VIT）
  - 1D RoPE position_ids（无 3D MRoPE）
  - 使用 AutoTokenizer（无 AutoProcessor）

用法:
  python eval/gen/gen_images_qwen3_mot.py \
      --output_dir ./output --prompt "A red ball on the ground" \
      --model-path /path/to/mot_ckpt --llm-path /path/to/qwen3-8b --vae-path /path/to/ae.safetensors
"""

import os
import json
import argparse
import time
import random

import numpy as np
import torch
from safetensors.torch import load_file
from PIL import Image

from transformers import AutoTokenizer

# === Qwen3 text-only MOT (Mixture-of-Transformers) 模型 import ===
from modeling.unimm.unimm_mot import UnimmMoT, UnimmConfig
from modeling.unimm.qwen3_mot import Qwen3MoTForConditionalGeneration
from modeling.unimm.qwen3 import NaiveCache
from modeling.qwen3.configuration_qwen3 import Qwen3Config
from modeling.autoencoder import load_ae

import torch._dynamo
torch._dynamo.config.disable = True


def get_device_info():
    """检测设备类型并返回相关信息（NPU/CUDA 双设备支持）"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu"
        backend = "hccl"
        device_count = torch.npu.device_count()
        def set_device(idx): torch.npu.set_device(idx)
        def get_device_name(): return torch.npu.get_device_name()
        def synchronize(): torch.npu.synchronize()
        def empty_cache(): torch.npu.empty_cache()
    elif torch.cuda.is_available():
        device_type = "cuda"
        backend = "nccl"
        device_count = torch.cuda.device_count()
        def set_device(idx): torch.cuda.set_device(idx)
        def get_device_name(): return torch.cuda.get_device_name()
        def synchronize(): torch.cuda.synchronize()
        def empty_cache(): torch.cuda.empty_cache()
    else:
        raise RuntimeError("No CUDA or NPU device available")

    return {
        'type': device_type,
        'backend': backend,
        'device_count': device_count,
        'set_device': set_device,
        'get_device_name': get_device_name,
        'synchronize': synchronize,
        'empty_cache': empty_cache,
    }


def move_generation_input_to_device(generation_input, device):
    """将 generation_input dict 中所有 tensor 搬到指定 device"""
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def generate_image(
    prompt,
    num_timesteps=50,
    cfg_scale=10.0,
    cfg_interval=None,
    cfg_renorm_min=0.,
    timestep_shift=1.0,
    num_images=4,
    resolution=512,
    inference_mode='flash',
    device=None,
    device_type='cuda',
    gen_model=None,
    tokenizer=None,
    new_token_ids=None,
    vae_model=None,
    seed=None,
):
    if cfg_interval is None:
        cfg_interval = [0, 1.0]

    # 创建独立的随机数生成器，确保初始噪声可复现且与外部 RNG 状态无关
    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

    # Qwen3 text-only: num_hidden_layers 直接在 llm_config 上（flat config）
    num_layers = gen_model.config.llm_config.num_hidden_layers
    past_key_values = NaiveCache(num_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    # 1. 编码 text prompt
    generation_input_text, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * num_images,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_text = move_generation_input_to_device(generation_input_text, device)

    with torch.no_grad():
        with torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input_text)

    # 2. 准备 VAE latent 输入 (Qwen3 text-only: 1D position_ids)
    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
        new_token_ids=new_token_ids,
        generator=generator,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    # 3. 准备 CFG（无条件）输入 (Qwen3 text-only: 1D position_ids)
    # 训练时 CFG dropout 只丢弃 user\n{caption} 段，但始终保留 [bos]assistant\n[eos]
    # 因此无条件路径的 KV cache 中必须包含 assistant 前缀
    cfg_past_key_values = NaiveCache(num_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    # 编码 assistant\n 前缀到 CFG KV cache
    cfg_prompt_input, cfg_newlens, cfg_new_rope = gen_model.prepare_cfg_prompts(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        num_images=num_images,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    cfg_prompt_input = move_generation_input_to_device(cfg_prompt_input, device)

    with torch.no_grad():
        with torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
            cfg_past_key_values = gen_model.forward_cache_update_text(
                cfg_past_key_values, **cfg_prompt_input)

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    # 4. 生成 image latent
    with torch.no_grad():
        with torch.amp.autocast(device_type, enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                generation_input_text=generation_input_text,
                inference_mode=inference_mode,
                **generation_input,
            )

    # 5. VAE decode
    latent_ch = gen_model.config.vae_config.z_channels
    latent_ds = gen_model.config.vae_config.downsample
    p = gen_model.config.latent_patch_size
    image_list = []
    for latent in unpacked_latent:
        h, w = resolution // latent_ds // p, resolution // latent_ds // p
        latent = latent.reshape(1, h, w, p, p, latent_ch)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, latent_ch, h * p, w * p)
        latent = latent.to(device=device, dtype=torch.float32)
        image = vae_model.decode(latent)
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


def load_model(args, device, device_type='cuda'):
    """加载 Qwen3 text-only MOT Packed SDPA 模型和相关组件"""
    llm_config = Qwen3Config.from_pretrained(args.llm_path)
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    # 设置 SDPA 注意力实现
    llm_config._attn_implementation = 'sdpa'

    vae_model, vae_config = load_ae(local_path=args.vae_path, vae_type=getattr(args, 'vae_type', 'flux1-vae'))

    config = UnimmConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
        use_qwen_vit=False,  # Qwen3 text-only: 不使用 Qwen VIT
        use_moe=True,
    )

    # 手动实例化（绕过 from_pretrained 的 meta-device 问题），
    # 然后直接加载训练 checkpoint（已包含完整权重，无需先加载 HF 原始权重）
    import gc

    print("  [1/4] Instantiating Qwen3MoTForConditionalGeneration on CPU (bfloat16)...")
    language_model = Qwen3MoTForConditionalGeneration.from_pretrained(
        args.llm_path, config=llm_config,
        key_mapping=Qwen3MoTForConditionalGeneration._checkpoint_conversion_mapping)
    print(f"  [1/4] Done. Parameters: {sum(p.numel() for p in language_model.parameters()) / 1e9:.2f}B")

    print("  [2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

    print("  [3/4] Building UnimmMoT model...")
    model = UnimmMoT(language_model, None, tokenizer, config)

    new_token_ids = {
        'bos_token_id': tokenizer.encode("<|im_start|>")[0],
        'eos_token_id': tokenizer.encode("<|im_end|>")[0],
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
        'start_of_image': tokenizer.encode("<|vision_start|>")[0],
        'end_of_image': tokenizer.encode("<|vision_end|>")[0],
    }

    print("  [4/4] Loading training checkpoint (contains full model weights)...")
    model_state_dict_path = os.path.join(args.model_path, "model.safetensors")
    if os.path.exists(model_state_dict_path):
        model_state_dict = load_file(model_state_dict_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(f"  [4/4] Checkpoint loading: {msg}")
        del model_state_dict
    else:
        print(f"  [4/4] Warning: model.safetensors not found in {args.model_path}")
    gc.collect()

    model = model.to(device).to(dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device).to(dtype=torch.float32).eval()

    return model, tokenizer, new_token_ids, vae_model


# ═══════════════════════════════════════════════════════
# MeiGen 测试用例
# ═══════════════════════════════════════════════════════
MEIGEN_TEST_PROMPTS = [
    # 1. 中文电商海报 (Chinese E-commerce Poster)
    "A festive red and gold promotional poster for Chinese New Year. In the center, a 3D rendered gift box is open, revealing cosmetics products. Large, bold golden Chinese characters at the top read '新春大促' (New Year Sale). Below it, smaller white text says '全场五折起' (50% off all items). The background features traditional cloud patterns and lanterns. A button at the bottom reads '立即抢购' (Buy Now).",

    # 2. 中文新闻网页截图 (Chinese News Website)
    "A screenshot of a Chinese news portal on a mobile screen. The header is red with the white logo '每日新闻' (Daily News). The main headline reads '科技创新引领未来' in bold black Songti font. Below the headline is a photo of a modern laboratory with researchers in white coats. Under the photo, a caption reads '某科技园区的研发中心'. The UI includes a bottom navigation bar with icons and text: '首页', '视频', '我的'.",

    # 3. 中文美食社交分享 (Chinese Food Social Media)
    "A high-angle photo of a delicious bowl of beef noodle soup, styled for Xiaohongshu (Little Red Book). The noodles are topped with cilantro and chili oil. Hand-drawn style white text overlay says '超好吃的牛肉面!' (Super delicious beef noodles!) with a cute heart sticker. The background is a wooden table with chopsticks and a vintage soda bottle. The lighting is warm and cozy.",

    # 4. 中文房地产广告 (Chinese Real Estate Ad)
    "A luxury real estate advertisement flyer. The main image shows a high-rise apartment building at night with lights on. A semi-transparent blue box on the left contains vertical white text: '江景豪宅' (River View Mansion) in a calligraphy style. Below it, '尊贵生活，从此开始'. In the bottom right corner, a gold badge says 'VIP 预约中'.",

    # 5. 中文古诗词配图 (Chinese Poetry Illustration)
    "An ink wash painting style image illustrating a Chinese poem. A solitary boat floats on a misty river near mountains. Vertical traditional Chinese text on the right reads: '孤舟蓑笠翁，独钓寒江雪'. The mood is serene and melancholic. A red seal stamp (chop) is visible in the bottom left corner.",

    # 6. 日文动漫杂志封面 (Japanese Anime Magazine)
    "A colorful magazine cover titled 'ANIME WORLD'. The central character is a magical girl with pink hair holding a wand. Large, bold Katakana text spreads across the image: '新作アニメ特集' (New Anime Feature). A burst bubble in the corner says '完全保存版' (Complete Collector's Edition). The background is filled with star patterns and vibrant colors.",

    # 7. 日文居酒屋菜单 (Japanese Izakaya Menu)
    "A close-up of a handwritten wooden menu board in a Japanese Izakaya. The wood texture is aged. Vertical black brush calligraphy lists items: '焼き鳥' (Yakitori), '刺身盛り合わせ' (Sashimi Platter), and '生ビール' (Draft Beer). Next to the text are small, simple illustrations of a beer mug and a chicken skewer. The lighting is dim and moody.",

    # 8. 韩文护肤品广告 (Korean Skincare Ad)
    "A clean, minimalist advertisement for a moisturizer. A bottle of white cream sits on a reflective water surface. The background is pale blue. Elegant Korean text in the center reads: '수분 가득한 피부' (Skin full of moisture). Below it, smaller English text says 'Hydro Boost'. A 'Best Seller' badge is in the top right corner.",

    # 9. 多语言机场指示牌 (Multilingual Airport Sign)
    "A realistic photo of a modern airport terminal sign hanging from the ceiling. The sign has a yellow background with black text. It shows directions with arrows. The text is in three languages: English 'Departures', Chinese '出发', and Japanese '出発'. An icon of an airplane taking off is next to the text. The background shows a blurred terminal hall with travelers.",

    # 10. 混合语言街头招牌 (Mixed Language Street Sign - Hong Kong style)
    "A neon-lit street scene in Hong Kong at night. A prominent vertical neon sign reads '金龙餐厅' (Golden Dragon Restaurant) in red traditional Chinese characters. Below it, a smaller horizontal sign says 'OPEN 24 HOURS' in English. Another sign in the background has '海鲜' (Seafood) in green neon. The atmosphere is cyberpunk with wet pavement reflecting the lights.",

    # 1. 新闻摄影 (News Photography): 检查是否能生成具有新闻现场感、多人同框且背景复杂的真实照片
    "A photo of a press conference with microphones and a blurred crowd in the background, realistic news style",

    # 2. 网页/UI设计 (Web UI): 检查模型对矩形、按钮、排版布局的理解
    "A screenshot of a modern landing page for a tech startup, clean UI, blue and white color scheme",

    # 3. 社交媒体帖子 (Social Media Content): 检查常见的自拍/生活分享风格，包含滤镜感
    "An Instagram photo of a latte art coffee and a croissant on a cafe table, top-down view, lifestyle filter",

    # 4. 电商产品图 (E-commerce): 检查白底图、产品质感及简单的促销文字
    "A professional product photo of a sneaker on a white background, studio lighting, advertising style",

    # 5. 信息图表/PPT (Infographic): 检查对图表、箭头、数据可视化的生成能力（通抓数据中很常见）
    "An infographic showing a bar chart with colorful bars and some text statistics, business presentation style",

    # 6. 街拍/纪实 (Street Snap): 检查复杂的街道背景、路人、招牌和真实光影
    "A candid street photography shot of people walking in busy Tokyo, daytime, realistic lighting",

    # 7. 论坛/博客文章配图 (Blog Header): 检查常见的抽象概念配图，如“科技与自然”的结合
    "A digital illustration for a blog post about artificial intelligence, connecting brain circuits with a tree",

    # 8. 扫描件/文档 (Document/Scan): 检查模型是否学到了报纸、海报或扫描文档的纹理（通抓数据中可能有旧资料）
    "A vintage newspaper front page with headlines and black and white photos, slightly grainy texture",

    # 9. 房地产/室内设计 (Real Estate): 检查广角镜头、室内透视和家居细节
    "A wide-angle photo of a modern living room with a large window and beige sofa, real estate listing style",

    # 10. 文字密集型海报 (Poster with Text): 挑战 Flux 的强项，检查在复杂背景下的文字排版能力
    "A movie poster design with the title 'THE FUTURE' in bold letters, dark sci-fi background",

    # 1. 基础人像 (Base Human): 检查面部结构是否扭曲，肤色是否过饱和（烧焦）
    "A close-up photo of a girl, smiling, natural lighting, high quality",

    # 2. 基础物体 & 色彩 (Simple Object & Color): 检查颜色溢出，如果苹果是荧光红，说明 LR 太大
    "A red apple on a wooden table, cinematic lighting",

    # 3. 动物 & 纹理 (Texture): 检查毛发细节。如果是一团糊，说明还在早期；如果是噪点，说明崩了
    "A cute white cat sitting on a sofa",

    # 4. 复杂场景 & 光影 (Complex Scene): 检查构图和高频细节
    "A futuristic cyberpunk city street at night, neon lights, rain, reflection",

    # 5. 抽象/风格 (Style): 检查模型是否只记住了照片，还是学会了风格
    "An oil painting of a mountain landscape at sunset",

    # 6. 文字能力 (Flux特长): 如果能画出文字，说明语义对齐很好
    "A sign board that says 'HELLO WORLD'",

    "A red ball on the ground",
    "A beautiful beach with golden sand and blue ocean waves",
    "A dense green forest with sunlight streaming through the trees",
    "A snowy mountain peak under a starry night sky",
    "A desert landscape during a dramatic red sunset",
    "A long straight road leading to a mountain in the distance",
    "A waterfall cascading down a cliff into a pool",
    "A watercolor painting of a rainy city street",

    # 1. 复杂新闻现场 (News & Journalism)
    "A wide-angle, high-resolution shot of a chaotic press briefing outdoors. In the foreground, a diverse group of reporters holds microphones with various news channel logos (like 'CNN', 'BBC', 'FOX') towards a podium. Behind the podium stands a middle-aged politician in a navy suit, gesturing with his right hand, looking serious. The background shows a government building with white columns, partially obscured by a crowd of onlookers. The lighting is natural overcast daylight, creating soft shadows. A chyron banner at the bottom of the frame reads 'BREAKING NEWS: POLICY UPDATE' in bold white text on a red background.",

    # 2. 网页着陆页 (Web Landing Page UI)
    "A full-screen screenshot of a sleek, modern SaaS landing page. The header features a bold sans-serif title 'Automate Your Workflow' in dark blue, centered. Below it, a subheadline says 'Save time and boost productivity with AI-driven tools.' A primary call-to-action button in bright green with the text 'Get Started Free' is positioned centrally. On the right side of the hero section, there is an isometric 3D illustration of a robot arm organizing colorful data blocks. The navigation bar at the top is white with a logo on the left and menu items 'Features', 'Pricing', 'Blog' on the right.",

    # 3. 电商产品详情页 (E-commerce Product Detail)
    "A vertical promotional banner for a skincare product. The background is a soft pastel peach gradient. In the center, a realistic amber glass bottle with a black dropper cap is labeled 'Radiance Serum' in elegant serif font. Surrounding the bottle are fresh ingredients: slices of orange, a splash of water, and green leaves, arranged dynamically to suggest freshness. At the bottom, a testimonial quote in italics reads 'Best serum I have ever used! - Jane Doe' followed by a 5-star rating graphic. The lighting is bright and studio-quality with sharp reflections on the glass bottle.",

    # 4. 科技博客文章头图 (Tech Blog Header)
    "A conceptual digital illustration for a blog post about 'Cybersecurity in 2025'. The composition features a glowing blue digital padlock in the center, surrounded by a network of circuit board lines connecting to icons of a laptop, a smartphone, and a cloud server. The background is a deep matrix-green with falling binary code rain effects. In the top left corner, there is a semi-transparent text overlay 'TechInsights' in a futuristic font. The style is flat vector art with vibrant neon gradients.",

    # 5. 房地产房源展示 (Real Estate Listing)
    "A professionally photographed interior shot of a luxury open-concept living room. The room features floor-to-ceiling windows on the left, revealing a cityscape view at dusk. A large L-shaped beige sectional sofa sits on a textured grey rug. On the coffee table, there is a vase of white tulips and a stack of magazines. To the right, a modern kitchen island with marble countertops and three bar stools is visible. The image has a watermark in the bottom right corner reading 'Luxury Estates'. The lighting is warm, combining recessed ceiling lights and the blue hour natural light from the window.",

    # 6. 学术会议海报 (Academic Conference Poster)
    "A layout design for a scientific conference poster titled 'Advances in Quantum Computing'. The poster is divided into three columns. The left column contains an abstract text block and a list of authors. The middle column features a large, detailed diagram of a qubit processor with labels 'Control Lines' and 'Readout'. The right column shows two line graphs with blue and red data points, titled 'Error Rates' and 'Coherence Time'. The color scheme is professional navy blue and white. The header has the logos of two universities.",

    # 7. 社交媒体食谱分享 (Social Media Recipe Card)
    "A top-down, flat-lay photograph designed for Pinterest. The center features a rustic wooden cutting board with a finished dish: a grilled salmon fillet with asparagus. Surrounding the main dish are raw ingredients: a lemon cut in half, a small bowl of sea salt, fresh dill, and a bottle of olive oil. Overlaid text in a handwritten white font says '15-Minute Healthy Dinner'. Arrows drawn in a doodle style point to the ingredients with labels like 'Fresh Lemon' and 'Omega-3'. The lighting is bright and airy.",

    # 8. 财经新闻图表 (Financial News Infographic)
    "A digital infographic illustrating stock market trends. The background is a dark grey grid. A jagged green line graph trends upwards from left to right, with peak points labeled '$150', '$200', and '$280'. Below the graph, a bar chart shows trading volume in blue bars. On the right side, a text box summarizes 'Key Takeaways' with bullet points. The headline at the top reads 'Market Rally Continues' in bold uppercase letters. A stock ticker tape runs along the very bottom with symbols like 'AAPL +1.2%', 'GOOG -0.5%'.",

    # 9. 旅游杂志封面 (Travel Magazine Cover)
    "A magazine cover design for 'Wanderlust'. The main image is a breathtaking shot of Santorini, Greece, showing white buildings with blue domes on a cliffside overlooking the Aegean Sea. The sky is a clear azure blue. The magazine title 'Wanderlust' spans the top in a large, white, serif typeface. Cover lines on the left include '10 Hidden Gems in Europe', 'Budget Travel Tips', and 'Best Summer Eats'. A barcode and price tag '$5.99' are in the bottom left corner.",

    # 10. 软件教程截图 (Software Tutorial Screenshot)
    "A simulated screenshot of a video editing software interface. The UI is dark mode with a timeline track at the bottom containing colored clips (blue for video, green for audio). The top left panel shows a preview window of a dog running in a park. The top right panel is an inspector showing sliders for 'Exposure', 'Contrast', and 'Saturation'. A bright yellow cursor arrow points to the 'Export' button in the top right corner, simulating a tutorial guide. Text overlay says 'Step 1: Adjust Colors'.",

    # --- 1. 复杂光影与材质 (Lighting & Material) ---
    # 检查模型对透明、反光、次表面散射等物理属性的理解
    "A crystal glass apple on a black velvet table, caustic light patterns reflecting on the surface, dramatic rim lighting.",
    "A close-up of a human eye, extreme detail, reflection of a city skyline in the iris, macro photography.",
    "A statue made of translucent jade, glowing from within, set in a misty ancient temple, soft ethereal lighting.",
    "Chrome sphere floating in a desert, reflecting the warped horizon and blue sky, ray-tracing style.",

    # --- 2. 艺术风格迁移 (Artistic Styles) ---
    # 检查模型对不同艺术流派的掌握
    "A portrait of a cat in the style of Vincent van Gogh's 'The Starry Night', swirling brushstrokes, blue and yellow color palette.",
    "A cyberpunk city street rendered in Ukiyo-e woodblock print style, flat colors, textured paper effect.",
    "A futuristic robot drawn in the style of Leonardo da Vinci's sketches, sepia paper, ink lines, anatomical details.",
    "A landscape of a candy kingdom in 8-bit pixel art style, retro game aesthetic.",
    "A minimalistic line art drawing of a single rose, continuous line, white background.",

    # --- 3. 空间与透视 (Spatial & Perspective) ---
    # 检查模型对空间关系的理解，如俯视、鱼眼、等轴测
    "An isometric view of a cozy gamer's bedroom, cutaway style, showing a desk with three monitors, a bed, and shelves with figurines.",
    "A fish-eye lens view of a skater jumping over a staircase, distortion effect, dynamic action shot.",
    "A drone shot looking straight down at a busy intersection in New York, zebra crossings, yellow taxis, organized chaos.",
    "A view from inside a refrigerator looking out, shelves stocked with colorful food, door open, kitchen background visible.",

    # --- 4. 抽象与超现实 (Abstract & Surreal) ---
    # 检查模型的想象力和非逻辑组合能力
    "A giant clock melting over a tree branch in a desert, Salvador Dali style, dreamlike atmosphere.",
    "An astronaut floating in an ocean of colorful jellybeans, galaxy background, surreal composition.",
    "A staircase leading up into clouds, with doors floating in the sky, magical realism.",
    "A portrait of a woman whose hair is made of flowing water and flowers, fantasy concept art.",

    # --- 5. 复杂语义与计数 (Counting & Semantics) ---
    # 检查模型是否能准确理解数量和特定位置关系 (这是 Diffusion 模型的难点)
    "Three red apples and two green pears arranged in a circle on a wooden table.",
    "A photo of a hand holding exactly four playing cards: Ace, King, Queen, and Jack of Spades.",
    "A square blue box on top of a round red table, with a yellow triangle rug underneath.",
    "A row of five rubber ducks, colored from left to right: red, orange, yellow, green, blue.",

    # --- 6. 文本渲染挑战 (Advanced Text Rendering) ---
    # 检查模型对非标准字体、长文本、融合文本的处理
    "A neon sign on a brick wall that says 'NEVER GIVE UP' in pink cursive script.",
    "A blackboard with a chalk drawing of a coffee cup and the handwritten text 'Menu: Espresso $2, Latte $3'.",
    "A vintage postcard with the text 'Greetings from Paris' written in large 3D retro lettering with scenes of the Eiffel Tower inside the letters.",
    "A t-shirt design with the text 'CODE IS POETRY' printed in a glitchy, hacker-style font.",

    # --- 7. 人物多样性与动作 (Diversity & Action) ---
    # 检查模型对不同人种、年龄、复杂动作的生成
    "An elderly Asian man playing chess in a park, focused expression, wrinkles visible, natural lighting.",
    "A young African American woman dancing ballet in a studio, leaping in the air, dynamic pose, motion blur.",
    "A group of diverse friends (Caucasian, Hispanic, Asian) laughing and toasting with drinks at a rooftop party at sunset.",
    "A close-up of a construction worker wearing a yellow helmet, face covered in dust, gritty realistic texture.",

    # --- 8. 动物与生物 (Animals & Creatures) ---
    "A majestic lion wearing a king's crown and a red royal cloak, sitting on a throne, fantasy oil painting.",
    "A macro shot of a colorful chameleon on a green leaf, scales texture highly detailed, bokeh background.",
    "A hybrid creature that is half owl and half eagle, flying in the sky, mythical beast concept art.",
    "A group of penguins waddling on an iceberg, realistic documentary style, 4k.",

    # --- 9. 中国文化元素 (Chinese Culture Specifics) ---
    # 针对 MeiGen 模型的中文能力进行深度测试
    "A traditional Chinese ink painting of the Great Wall, misty mountains, black and white style.",
    "A photo of a steaming bamboo steamer basket filled with Xiao Long Bao (soup dumplings), appetizing food photography.",
    "A close-up of a Peking Opera mask, vibrant colors (red, black, white), intricate patterns.",
    "A festive scene of a Lion Dance performance during Lunar New Year, dynamic movement, red lanterns in background.",

    # --- 10. 长Prompt与细节描述 (Long Prompt & Detail Adherence) ---
    # 检查模型是否能遵循复杂的长指令
    "A highly detailed interior of a wizard's library. There are tall wooden bookshelves filled with old leather-bound books. In the center, a large oak desk is covered with scrolls, a glowing crystal ball, and a sleeping owl. A fireplace is crackling on the right. A spiral staircase leads to a second level. The lighting is dim, coming from the fireplace and floating candles. Dust motes are dancing in the light beams.",

    "In the image is a male wearing traditional Arab attire. He dons a white robe and sports a headscarf with red and white stripes, accentuated by a black headband. His arms are crossed over his chest, and a smile adorns his face, exuding confidence and friendliness. The background is gray, lending an aura of simplicity and solemnity to the entire scene.Keywords: Arab traditional attire, white robe, red and white striped headscarf, black headband, crossed arms, smile, confidence, friendly, gray background, simple, solemn.",
    "A young woman with long black hair, holding a bouquet of fresh flowers, dressed in a summer white dress, stands in a park brimming with blooming flowers, with the sunlight casting its glow upon her face.",
    "A young girl and her puppy are interacting in the family garden, the girl laughing and stroking the dog's head, wearing a pink dress, amidst the fragrance of flowers all around.",
    "A middle-aged man, his face a little weary, dressed in a dark blue suit, holding a cup of espresso, standing on the balcony before the city skyline.",
    "A young woman with fair skin, her long blonde hair styled in an updo, clad in a pristine wedding gown, gently caresses rose petals. The backdrop is a charming garden bathed in white flowers.",
    "A well-built man wearing a red turban and a blue sports outfit is holding a basketball, standing on the schoolyard court for a shot.",
    "A young girl, with short black hair, dressed in a yellow dress, holds a small gray rabbit with both hands, and the background is a forest stream with sunlight filtering through the leaves.",
    "A young dancer stands on stage, focused, wearing a dazzling purple dancing outfit, raising their hands to interact with the sparkling chandelier of the ceiling, with stage lights focusing on them.",
    "An elderly person, with a smile on their face, wearing a gray sweater, seated on a park bench, holding a copy of the day's newspaper, with a small mixed breed dog lying beside them.",
    "A young man with slightly long hair, wearing a chef's hat and a white chef's uniform, stands in a modern kitchen, stirring with a spatula a colorful medley of vegetables.",
    "Traditional food of the Mid-Autumn Festival",
    "Show the most common outdoor activity associated with the Dragon Boat Festival",
]




def main():
    parser = argparse.ArgumentParser(description="Generate images using Qwen3 text-only MOT (Mixture-of-Transformers) model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to generate images from.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Text file containing one prompt per line.")
    parser.add_argument("--metadata_file", type=str, default=None, help="JSONL file with metadata for each prompt.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale.")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution of generated images.")
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--llm-path', type=str, default='hf/Qwen3-8B/')
    parser.add_argument('--vae-path', type=str, default=None)
    parser.add_argument('--vae-type', type=str, default='flux1-vae', choices=['flux1-vae', 'flux2-vae'],
                        help="VAE type: 'flux1-vae' (BAGEL, z=16) or 'flux2-vae' (z=32)")
    parser.add_argument('--inference-mode', type=str, default='flash',
                        help='Inference mode (e.g., standard, fast, quality, custom)')
    parser.add_argument('--inference-steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_meigen_test_cases", action="store_true",
                        help="Use the same test cases as MeiGen inference_t2i.py")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed multi-GPU inference (requires torchrun)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 设备检测
    # ------------------------------------------------------------------
    device_info = get_device_info()
    device_type = device_info['type']
    print(f"Using device type: {device_type}")

    # ------------------------------------------------------------------
    # 输入参数验证
    # ------------------------------------------------------------------
    input_sources = sum([
        args.prompt is not None,
        args.prompt_file is not None,
        args.metadata_file is not None,
        args.use_meigen_test_cases,
    ])
    if input_sources == 0:
        raise ValueError("必须指定 --prompt, --prompt_file, --metadata_file 或 --use_meigen_test_cases 其中之一")
    if input_sources > 1:
        raise ValueError("只能指定 --prompt, --prompt_file, --metadata_file, --use_meigen_test_cases 其中之一")

    # ------------------------------------------------------------------
    # 随机种子
    # ------------------------------------------------------------------
    seed = args.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif device_type == 'npu':
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # 设备 & 分布式
    # ------------------------------------------------------------------
    if args.distributed:
        import datetime, torch.distributed as dist
        dist.init_process_group(backend=device_info['backend'], timeout=datetime.timedelta(seconds=3600))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device_info['set_device'](local_rank)
        device = torch.device(f"{device_type}:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device_info['set_device'](0)
        device = torch.device(f"{device_type}:0")

    if rank == 0:
        print(f"Using device: {device}, world_size: {world_size}")

    # ------------------------------------------------------------------
    # 输出目录
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images will be saved in {output_dir}")

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    if rank == 0:
        print("Loading Qwen3 text-only MOT Packed SDPA model...")
    start_time = time.time()
    gen_model, tokenizer, new_token_ids, vae_model = load_model(args, device, device_type)
    if rank == 0:
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 准备 prompt 列表
    # ------------------------------------------------------------------
    prompts = []

    if args.use_meigen_test_cases:
        prompts = MEIGEN_TEST_PROMPTS
        if rank == 0:
            print(f"Using {len(prompts)} MeiGen test cases")

    elif args.prompt is not None:
        prompts = [args.prompt]
        if rank == 0:
            print(f"Using single prompt: '{args.prompt}'")

    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if rank == 0:
            print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    elif args.metadata_file is not None:
        with open(args.metadata_file, "r", encoding="utf-8") as fp:
            metadatas = [json.loads(line) for line in fp]
        prompts = [metadata.get('prompt', '') for metadata in metadatas]
        if rank == 0:
            print(f"Loaded {len(prompts)} prompts from {args.metadata_file}")

    # ------------------------------------------------------------------
    # 分配 prompt 到各 GPU
    # ------------------------------------------------------------------
    if world_size > 1:
        prompts_per_gpu = (len(prompts) + world_size - 1) // world_size
        start_idx = rank * prompts_per_gpu
        end_idx = min(start_idx + prompts_per_gpu, len(prompts))
        local_prompts = list(enumerate(prompts))[start_idx:end_idx]
        print(f"GPU {rank}: Processing {len(local_prompts)} prompts (indices {start_idx} to {end_idx - 1})")
    else:
        local_prompts = list(enumerate(prompts))

    # ------------------------------------------------------------------
    # 生成参数
    # ------------------------------------------------------------------
    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 1.0
    num_timesteps = args.inference_steps
    cfg_renorm_min = 0.9
    num_images_per_prompt = args.num_images_per_prompt
    batch_size = min(args.batch_size, num_images_per_prompt)

    if rank == 0:
        print(f"Start generating {len(prompts)} prompts (this GPU: {len(local_prompts)})...")
        print(f"Each prompt will generate {num_images_per_prompt} images")
        print(f"Batch size: {batch_size}")
        print(f"Resolution: {args.resolution}")
        print(f"CFG scale: {cfg_scale}")
        print(f"Inference steps: {num_timesteps}")

    total_start_time = time.time()

    # ------------------------------------------------------------------
    # 生成循环
    # ------------------------------------------------------------------
    for prompt_idx, prompt in local_prompts:
        print(f"[GPU {rank}] [{prompt_idx + 1}/{len(prompts)}] Generating: {prompt[:60]}...")

        # 创建 prompt 特定的输出目录
        safe_prompt = prompt.replace(" ", "_").replace("'", "").replace('"', '').replace("/", "_")[:50]
        prompt_output_dir = os.path.join(output_dir, f"prompt_{prompt_idx:03d}_{safe_prompt}")
        os.makedirs(prompt_output_dir, exist_ok=True)

        # 保存 prompt 信息
        with open(os.path.join(prompt_output_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

        # 检查是否已生成（断点续跑）
        existing_images = [f for f in os.listdir(prompt_output_dir) if f.endswith('.png')]
        if len(existing_images) >= num_images_per_prompt:
            print(f"  Skipping (already {len(existing_images)} images)")
            continue

        # 分批生成
        all_images = []
        for batch_idx in range(0, num_images_per_prompt, batch_size):
            current_batch_size = min(batch_size, num_images_per_prompt - batch_idx)
            print(f"  Batch {batch_idx // batch_size + 1}/"
                  f"{(num_images_per_prompt + batch_size - 1) // batch_size} "
                  f"({current_batch_size} images)")

            batch_start_time = time.time()

            try:
                image_list = generate_image(
                    prompt=prompt,
                    cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    num_images=current_batch_size,
                    resolution=args.resolution,
                    inference_mode=args.inference_mode,
                    device=device,
                    device_type=device_type,
                    gen_model=gen_model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    vae_model=vae_model,
                    seed=seed,
                )
                all_images.extend(image_list)

                batch_time = time.time() - batch_start_time
                print(f"    Batch completed in {batch_time:.2f}s "
                      f"({batch_time / current_batch_size:.2f} sec/image)")

            except Exception as e:
                print(f"    Error generating batch: {e}")
                import traceback
                traceback.print_exc()
                break

        # 保存图像
        for img_idx, image in enumerate(all_images):
            try:
                bbox = image.getbbox()
                if bbox:
                    image = image.crop(bbox)
            except Exception:
                pass
            image_path = os.path.join(prompt_output_dir, f"image_{img_idx:03d}.png")
            image.save(image_path)

        print(f"  Saved {len(all_images)} images to {prompt_output_dir}")

        # 清理显存
        device_info['empty_cache']()

    total_time = time.time() - total_start_time
    num_local = len(local_prompts)
    if num_local > 0:
        print(f"\n[GPU {rank}] Done! Generated {num_local} prompts in {total_time:.2f}s "
              f"(avg {total_time / num_local:.2f}s/prompt)")

    if args.distributed:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            print("All GPUs have completed.")


if __name__ == "__main__":
    main()
