import argparse
import os
import os.path as osp
import mmcv
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate a ReID dataset for tracking')
    parser.add_argument('base_path', help='base path where original data is stored')
    parser.add_argument('output_path', help='output path where ReID dataset will be stored')
    parser.add_argument('--val-split', type=float, default=0.2, help='validation split ratio')
    parser.add_argument('--min-object', type=int, default=1, help='minimum number of objects per identity')
    parser.add_argument('--max-object', type=int, default=50, help='maximum number of objects per identity')
    parser.add_argument('--vis-threshold', type=float, default=0.0, help='visibility threshold for objects')
    return parser.parse_args()

def generate_reid_dataset(args):
    base_path, output_path, val_split, min_object, max_object, vis_threshold = args.base_path, args.output_path, args.val_split, args.min_object, args.max_object, args.vis_threshold
    if not osp.isdir(output_path):
        os.makedirs(output_path)
    elif os.listdir(output_path):
        raise OSError(f'Directory must be empty: \'{output_path}\'')

    video_names = os.listdir(base_path)
    reid_train_folder = osp.join(output_path, 'imgs')
    os.makedirs(reid_train_folder, exist_ok=True)

    for video_name in tqdm(video_names):
        video_folder = osp.join(base_path, video_name)
        img_folder = osp.join(video_folder, 'img')
        data_file_path = osp.join(video_folder, 'gt/gt.txt')
        if not osp.exists(data_file_path):
            continue

        raw_img_names = sorted(os.listdir(img_folder))
        data_lines = mmcv.list_from_file(data_file_path)

        last_frame_id = -1
        for line in data_lines:
            line = line.strip().split(',')
            frame_id, ins_id = map(int, line[:2])
            x, y, w, h = map(float, line[2:6])
            confidence = float(line[6])

            if confidence < vis_threshold:
                continue

            reid_img_folder = osp.join(reid_train_folder, f'{video_name}_{ins_id:06d}')
            os.makedirs(reid_img_folder, exist_ok=True)
            idx = len(os.listdir(reid_img_folder))
            reid_img_name = f'{idx:06d}.jpg'
            if frame_id != last_frame_id:
                raw_img_name = raw_img_names[frame_id - 1]
                raw_img = mmcv.imread(f'{img_folder}/{raw_img_name}')
                last_frame_id = frame_id
            xyxy = np.asarray([x, y, x + w, y + h])
            reid_img = mmcv.imcrop(raw_img, xyxy)
            mmcv.imwrite(reid_img, f'{reid_img_folder}/{reid_img_name}')

    # Create training and validation lists
    split_data_into_train_val(output_path, reid_train_folder, val_split, min_object, max_object)

def split_data_into_train_val(output_path, reid_train_folder, val_split, min_object, max_object):
    reid_meta_folder = osp.join(output_path, 'meta')
    os.makedirs(reid_meta_folder, exist_ok=True)
    reid_train_list, reid_val_list, reid_entire_dataset_list = [], [], []
    reid_img_folder_names = sorted(os.listdir(reid_train_folder))
    num_ids = len(reid_img_folder_names)
    num_train_ids = int(num_ids * (1 - val_split))
    train_label, val_label = 0, 0

    random.seed(0)

    for reid_img_folder_name in reid_img_folder_names[:num_train_ids]:
        reid_img_names = os.listdir(osp.join(reid_train_folder, reid_img_folder_name))
        if len(reid_img_names) < min_object:
            continue
        if len(reid_img_names) > max_object:
            reid_img_names = random.sample(reid_img_names, max_object)
        for reid_img_name in reid_img_names:
            reid_train_list.append(f'{reid_img_folder_name}/{reid_img_name} {train_label}\n')
        train_label += 1

    for reid_img_folder_name in reid_img_folder_names[num_train_ids:]:
        reid_img_names = os.listdir(osp.join(reid_train_folder, reid_img_folder_name))
        if len(reid_img_names) < min_object:
            continue
        if len(reid_img_names) > max_object:
            reid_img_names = random.sample(reid_img_names, max_object)
        for reid_img_name in reid_img_names:
            reid_val_list.append(f'{reid_img_folder_name}/{reid_img_name} {val_label}\n')
            reid_entire_dataset_list.append(f'{reid_img_folder_name}/{reid_img_name} {train_label + val_label}\n')
        val_label += 1

    with open(osp.join(reid_meta_folder, f'train_{int(100 * (1 - val_split))}.txt'), 'w') as f:
        f.writelines(reid_train_list)
    with open(osp.join(reid_meta_folder, f'val_{int(100 * val_split)}.txt'), 'w') as f:
        f.writelines(reid_val_list)
    with open(osp.join(reid_meta_folder, 'train.txt'), 'w') as f:
        f.writelines(reid_entire_dataset_list)

if __name__ == '__main__':
    args = parse_args()
    generate_reid_dataset(args)
