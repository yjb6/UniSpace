import argparse
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default='./dpg_bench/dpg_bench.csv',
    )
    parser.add_argument(
        "--res-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pic-num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vqa-model",
        type=str,
        default='mplug',
    )

    args = parser.parse_args()
    return args

CKPT = os.path.join(
    os.environ.get("CKPT_ROOT", ""),
    "damo",
    "mplug_visual-question-answering_coco_large_en",
)

class MPLUG(torch.nn.Module):
    def __init__(self, ckpt=CKPT, device='gpu'):
        super().__init__()
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(
                "DPG-Bench mPLUG checkpoint was not found. Set CKPT_ROOT or "
                "pass a valid scorer checkpoint."
            )
        # Import mPLUG directly. ModelScope's generic VQA pipeline imports the
        # unrelated OFA/fairseq audio stack, which DPG does not use.
        from modelscope.models.multi_modal.mplug import MPlugConfig
        from modelscope.models.multi_modal.mplug_for_all_tasks import MPlugForAllTasks
        from modelscope.utils.constant import Tasks
        from transformers import BertTokenizer
        from torchvision import transforms

        self.device = torch.device(device)
        self.model = MPlugForAllTasks(
            ckpt, task=Tasks.visual_question_answering).to(self.device).eval()
        self.tokenizer = BertTokenizer.from_pretrained(ckpt)
        config = MPlugConfig.from_yaml_file(osp.join(ckpt, 'config.yaml'))
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (config.image_res, config.image_res),
                interpolation=Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def vqa(self, image, question):
        image_tensor = self.image_transform(image.convert('RGB')).unsqueeze(0)
        question_tokens = self.tokenizer(
            question.lower(), padding='max_length', truncation=True,
            max_length=25, return_tensors='pt')
        inputs = {
            'image': image_tensor.to(self.device),
            'question': question_tokens.to(self.device),
        }
        with torch.no_grad():
            result = self.model(inputs)
        return result['text']

def prepare_dpg_data(args):
    previous_id = ''
    current_id = ''
    question_dict = dict()
    category_count = defaultdict(int)
    # 'item_id', 'text', 'keywords', 'proposition_id', 'dependency', 'category_broad', 'category_detailed', 'tuple', 'question_natural_language'
    data = pd.read_csv(args.csv)
    for i, line in data.iterrows():
        if i == 0:
            continue

        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = line.dependency.split(',')
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]['qid2tuple'][qid] = line.tuple
            question_dict[current_id]['qid2dependency'][qid] = dependency_list_int
            question_dict[current_id]['qid2question'][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language})

        category = line.question_natural_language.split('(')[0].strip()
        category_count[category] += 1

        previous_id = current_id

    return question_dict

def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image

    cropped_image = input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))

    return cropped_image

def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
    generated_image = Image.open(image_path)
    crop_tuples_list = [
        (0,0,resolution,resolution),
        (resolution, 0, resolution*2, resolution),
        (0, resolution, resolution, resolution*2),
        (resolution, resolution, resolution*2, resolution*2),
    ]

    crop_tuples = crop_tuples_list[:args.pic_num]
    key = osp.basename(image_path).split('.')[0]
    value = question_dict.get(key, None)
    qid2tuple = value['qid2tuple']
    qid2question = value['qid2question']
    qid2dependency = value['qid2dependency']

    qid2answer = dict()
    qid2scores = dict()
    qid2validity = dict()
    detail_rows = []

    scores = []
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        for id, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2answer[id] = answer
            qid2scores[id] = float(answer == 'yes')
            detail_rows.append(
                image_path + ', ' + str(crop_tuple) + ', '
                + question + ', ' + answer + '\n')
        qid2scores_orig = qid2scores.copy()

        for id, parent_ids in qid2dependency.items():
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)
    average_score = sum(scores) / len(scores)
    result_row = (
        image_path + ', ' + ', '.join(str(i) for i in scores)
        + ', ' + str(average_score) + '\n')

    return average_score, qid2tuple, qid2scores_orig, result_row, detail_rows


def main():
    args = parse_args()

    print(f'DPG scorer Python: {sys.executable}')

    accelerator = Accelerator()

    question_dict = prepare_dpg_data(args)

    # Validate the complete benchmark snapshot before loading the large VQA
    # model or truncating any previous result file.
    image_suffixes = {'.jpg', '.jpeg', '.png', '.webp'}
    filename_list = sorted(
        fn for fn in os.listdir(args.image_root_path)
        if osp.isfile(osp.join(args.image_root_path, fn))
        and osp.splitext(fn)[1].lower() in image_suffixes
    )
    expected_keys = {str(key) for key in question_dict}
    image_keys = [osp.splitext(fn)[0] for fn in filename_list]
    actual_keys = set(image_keys)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if (len(image_keys) != len(actual_keys) or missing or extra):
        raise RuntimeError(
            f'DPG image coverage mismatch: images={len(image_keys)} '
            f'expected={len(expected_keys)} missing={missing[:20]} '
            f'extra={extra[:20]}'
        )

    timestamp = time.time()
    time_array = time.localtime(timestamp)
    time_style = time.strftime("%Y%m%d-%H%M%S", time_array)
    if args.res_path is None:
        args.res_path = osp.join(args.image_root_path, f'dpg-bench_{time_style}_results.txt')
    if os.path.isdir(args.res_path):
        args.res_path = os.path.join(args.res_path, "result_dpgbench.txt")
    if accelerator.is_main_process:
        with open(args.res_path, 'w') as f:
            pass
        with open(args.res_path.replace('.txt', '_detail.txt'), 'w') as f:
            pass

    device = str(accelerator.device)
    print("device:", device)
    if args.vqa_model == 'mplug':
        vqa_model = MPLUG(device=device)
    else:
        raise NotImplementedError
    vqa_model = accelerator.prepare(vqa_model)
    vqa_model = getattr(vqa_model, 'module', vqa_model)
    num_each_rank = len(filename_list) / accelerator.num_processes
    local_rank = accelerator.process_index
    local_filename_list = filename_list[round(local_rank * num_each_rank) : round((local_rank + 1) * num_each_rank)]

    local_scores = []
    local_category2scores = defaultdict(list)
    local_result_rows = []
    local_detail_rows = []
    model_id = osp.basename(args.image_root_path)
    print(f'Start to conduct evaluation of {model_id}')
    for fn in tqdm(local_filename_list):
        image_path = osp.join(args.image_root_path, fn)
        try:
            # compute score of one sample
            score, qid2tuple, qid2scores, result_row, detail_rows = compute_dpg_one_sample(
                args=args, question_dict=question_dict, image_path=image_path, vqa_model=vqa_model, resolution=args.resolution)
            local_scores.append(score)
            local_result_rows.append(result_row)
            local_detail_rows.extend(detail_rows)

            # summarize scores by categoris
            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split('(')[0].strip()
                qid_score = qid2scores[qid]
                local_category2scores[category].append(qid_score)

        except Exception as e:
            raise RuntimeError(f'Failed to score DPG image {fn}: {e}') from e

    accelerator.wait_for_everyone()
    global_dpg_scores = gather_object(local_scores)
    global_result_rows = gather_object(local_result_rows)
    global_detail_rows = gather_object(local_detail_rows)
    if len(global_dpg_scores) != len(filename_list):
        raise RuntimeError(
            f'DPG score coverage mismatch: scores={len(global_dpg_scores)} '
            f'images={len(filename_list)}')
    if not np.isfinite(global_dpg_scores).all() or not all(
            0.0 <= score <= 1.0 for score in global_dpg_scores):
        raise RuntimeError('DPG sample scores contain non-finite or out-of-range values')
    if len(global_result_rows) != len(filename_list):
        raise RuntimeError(
            f'DPG result-row coverage mismatch: rows={len(global_result_rows)} '
            f'images={len(filename_list)}')
    mean_dpg_score = np.mean(global_dpg_scores)

    global_categories = gather_object(list(local_category2scores.keys()))
    global_categories = set(global_categories)
    global_category2scores = dict()
    global_average_scores = []
    for category in global_categories:
        local_category_scores = local_category2scores.get(category, [])
        global_category2scores[category] = gather_object(local_category_scores)
        global_average_scores.extend(gather_object(local_category_scores))

    global_category2scores_l1 = defaultdict(list)
    for category in global_categories:
        l1_category = category.split('-')[0].strip()
        global_category2scores_l1[l1_category].extend(global_category2scores[category])

    time.sleep(3)
    if accelerator.is_main_process:
        # Only rank 0 writes shared artifacts. Concurrent append from all ranks
        # can leave a valid-looking but incomplete result file when one worker
        # exits early, and makes strict coverage validation unreliable.
        with open(args.res_path, 'w') as f:
            f.writelines(global_result_rows)
        with open(args.res_path.replace('.txt', '_detail.txt'), 'w') as f:
            f.writelines(global_detail_rows)

        output = f'Model: {model_id}\n'

        output += 'L1 category scores:\n'
        for l1_category in global_category2scores_l1.keys():
            output += f'\t{l1_category}: {np.mean(global_category2scores_l1[l1_category]) * 100}\n'

        output += 'L2 category scores:\n'
        for category in sorted(global_categories):
            output += f'\t{category}: {np.mean(global_category2scores[category]) * 100}\n'

        output += f'Image path: {args.image_root_path}\n'
        output += f'Save results to: {args.res_path}\n'
        output += f'DPG-Bench score: {mean_dpg_score * 100}'

        base_dir = os.path.dirname(args.res_path)
        score_path = os.path.join(base_dir, "result_dpgbench_score.txt")
        with open(score_path, 'w') as f:
            f.write(output + '\n')
        print(output)


if __name__ == "__main__":
    main()
