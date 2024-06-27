import cv2
import mmcv
import os
import numpy as np
import json
from mmtrack.apis import inference_mot, init_model
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time object tracking and labeling from video or webcam.')
    parser.add_argument('--input', help='Input video file path or webcam ID (integer).', default=0)
    parser.add_argument('--config', required=True, help='Path to the MOT model configuration file.')
    parser.add_argument('--detector-checkpoint', required=True, help='Path to the detector model checkpoint.')
    parser.add_argument('--reid-checkpoint', required=True, help='Path to the re-identification model checkpoint.')
    parser.add_argument('--json-path', required=True, help='Path to the JSON file containing size ranges.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for computation.')
    parser.add_argument('--log-file', default='process_log.txt', help='File to save processing logs.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for processing.')
    return parser.parse_args()


def draw_label_on_image(image, bbox, label):
    # Draw label on the image at bbox location
    x, y, w, h = bbox
    cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image
def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.detector.init_cfg.checkpoint = args.detector_checkpoint
    cfg.model.reid.init_cfg.checkpoint = args.reid_checkpoint

    # Load size ranges from JSON
    with open(args.json_path, 'r') as f:
        size_ranges = json.load(f)

    # Initialize model
    mot_model = init_model(cfg, device=args.device)

    # Set up video capture
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))  # Webcam ID
    else:
        cap = cv2.VideoCapture(args.input)  # Video file

    label_counts = {label: 0 for label in size_ranges}
    tracked_ids = {}
    frame_counter = 0
    start_time = time.time()

    # Open log file
    log_file = open(args.log_file, 'w')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = inference_mot(mot_model, frame, frame_id=frame_counter)
        for bbox in result['track_bboxes'][0]:
            if isinstance(bbox, np.ndarray) and bbox.ndim == 1 and len(bbox) >= 5:
                track_id, xmin, ymin, bbox_width, bbox_height = bbox[:5]
                diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
                label = "Undefined"
                for size_label, (min_diag, max_diag) in size_ranges.items():
                    if min_diag <= diagonal <= max_diag:
                        label = size_label
                        break

                if track_id not in tracked_ids.keys():
                    tracked_ids[track_id] = label
                    label_counts[label] += 1

                frame = draw_label_on_image(frame, (xmin, ymin, bbox_width, bbox_height), label)

        frame_counter += 1
        elapsed_time = time.time() - start_time
        real_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        log_file.write(f'Frame: {frame_counter}, Time: {real_time}, Labels: {label_counts}\n')
        print(f'Frame: {frame_counter}, Time: {real_time}, Labels: {label_counts}')

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

if __name__ == '__main__':
    main()
