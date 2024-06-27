import os
import os.path as osp
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset structure for object detection and tracking.")
    parser.add_argument('--base-path', required=True, help='Base path where the video folders are stored.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of the videos.')
    parser.add_argument('--width', type=int, default=1280, help='Width of the video frames.')
    parser.add_argument('--height', type=int, default=720, help='Height of the video frames.')
    parser.add_argument('--split-data', type=bool,default=False, help='Whether to split data into training and testing based on file naming.')
    return parser.parse_args()

def generate_dataset_structure(base_path, fps=30, width=1280, height=720, split_data=False):
    category_info = {"id": 1, "name": "coconat"}

    train_dataset = {
        "categories": [category_info],
        "annotations": [],
        "images": [],
        "videos": [],
        "num_instances": 0
    }

    test_dataset = {
        "categories": [category_info],
        "annotations": [],
        "images": [],
        "videos": [],
        "num_instances": 0
    }

    video_id = 0
    image_id = 0
    annotation_id = 0

    for video_name in os.listdir(base_path):
        video_path = osp.join(base_path, video_name)
        img_folder = osp.join(video_path, 'img')
        txt_file_path = osp.join(video_path, 'gt/gt.txt')

        if os.path.isdir(img_folder) and osp.exists(txt_file_path):
            video_id += 1
            video_info = {
                "id": video_id,
                "name": video_name,
                "fps": fps,
                "width": width,
                "height": height
            }

            is_test_set = video_name.endswith('4') and split_data
            if is_test_set:
                test_dataset["videos"].append(video_info)
            else:
                train_dataset["videos"].append(video_info)

            with open(txt_file_path, 'r') as f:
                frame_id = -1
                for line in f:
                    fields = line.strip().split(',')
                    frame_id_txt = int(fields[0])
                    if frame_id_txt != frame_id:
                        frame_id = frame_id_txt
                        image_id += 1
                        image_info = {
                            "id": image_id,
                            "video_id": video_id,
                            "file_name": osp.join(video_name, 'img', f"{frame_id:06d}.jpg").replace("\\", "/"),
                            "height": height,
                            "width": width,
                            "frame_id": frame_id - 1,
                            "mot_frame_id": frame_id
                        }

                        if is_test_set:
                            test_dataset["images"].append(image_info)
                        else:
                            train_dataset["images"].append(image_info)

                    annotation_id += 1
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "instance_id": int(fields[1]),
                        "bbox": [float(x) for x in fields[2:6]],
                        "area": float(fields[4]) * float(fields[5]),
                        "iscrowd": False,
                        "visibility": 1.0,
                        "mot_instance_id": int(fields[1]),
                        "mot_conf": float(fields[6]),
                        "mot_class_id": 1
                    }

                    if is_test_set:
                        test_dataset["annotations"].append(annotation_info)
                    else:
                        train_dataset["annotations"].append(annotation_info)

    train_dataset["num_instances"] = len(train_dataset["images"])
    test_dataset["num_instances"] = len(test_dataset["images"])

    with open('train_dataset.json', 'w') as train_file:
        json.dump(train_dataset, train_file, indent=4)

    with open('test_dataset.json', 'w') as test_file:
        json.dump(test_dataset, test_file, indent=4)

    print("Processing complete.")

if __name__ == '__main__':
    args = parse_args()
    generate_dataset_structure(args.base_path, args.fps, args.width, args.height, args.split_data)
