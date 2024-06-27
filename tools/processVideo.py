import argparse
import cv2
import mmcv
import os
import tempfile
import numpy as np
from mmtrack.apis import inference_mot, init_model

def parse_args():
    parser = argparse.ArgumentParser(description='Process video frames using MOT model and track objects.')
    parser.add_argument('--input-dir', required=True, help='Directory containing frames to process.')
    parser.add_argument('--mot-config-path', required=True, help='Path to the MOT model configuration file.')
    parser.add_argument('--detector-checkpoint', required=True, help='Path to the detector model checkpoint.')
    parser.add_argument('--reid-checkpoint', required=True, help='Path to the re-identification model checkpoint.')
    parser.add_argument('--output-video', required=True, help='Path where the output video will be saved.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of the output video.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for computation.')
    return parser.parse_args()

def process_frames(input_dir, mot_config, output_video, device='cuda:0', fps=30):
    if os.path.isdir(input_dir):
        frames = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')])
    elif os.path.isfile(input_dir) and input_dir.endswith(('.mp4', '.avi')):
        # Extract frames from video
        video = cv2.VideoCapture(input_dir)
        frames = []
        success, image = video.read()
        temp_dir = tempfile.TemporaryDirectory()
        frame_index = 1
        while success:
            frame_path = os.path.join(temp_dir.name, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(frame_path, image)
            frames.append(frame_path)
            success, image = video.read()
            frame_index += 1
        video.release()
    else:
        raise ValueError("Input path must be a directory of images or a video file.")


    num_frames = len(frames)

    mot_model = init_model(mot_config, device=device)
    prog_bar = mmcv.ProgressBar(num_frames)

    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name

    count = 0
    areas = []
    tracked_ids = set()

    for i, frame_file in enumerate(frames):
        img_path = os.path.join(input_dir, frame_file)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        mid_line_y = height // 2

        result = inference_mot(mot_model, img, frame_id=i)
        mot_model.show_result(
            img,
            result,
            show=False,
            wait_time=int(1000. / fps),
            out_file=f'{out_path}/{i:06d}.jpg')

        # Check for objects crossing the middle line
        for bbox in result['track_bboxes'][0]:
            if isinstance(bbox, np.ndarray) and bbox.ndim == 1 and len(bbox) >= 4:
                track_id,xmin, ymin, bbox_width, bbox_height = bbox[:5]
                xmax = xmin + bbox_width
                ymax = ymin + bbox_height
                if ymin <= mid_line_y <= ymax:
                    if track_id not in tracked_ids:
                      tracked_ids.add(track_id)
                      count += 1
                      bbox_area = bbox_width * bbox_height
                      areas.append(bbox_area)

        prog_bar.update()
    print(f'\nTotal objects crossed the line: {count}')

    print(f'\nMaking the output video at {output_video} with a FPS of {fps}')
    mmcv.frames2video(out_path, output_video, fps=fps, fourcc='mp4v')
    out_dir.cleanup()
    return f'\nTotal objects crossed the line: {count}'

if __name__ == '__main__':
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.mot_config_path)
    cfg.model.detector.init_cfg.checkpoint = args.detector_checkpoint
    cfg.model.reid.init_cfg.checkpoint = args.reid_checkpoint

    print(process_frames(args.input_dir, cfg, args.output_video, args.device, args.fps))