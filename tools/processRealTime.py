import argparse
import cv2
import mmcv
from mmtrack.apis import inference_mot, init_model

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time object tracking from webcam using an MOT model.')
    parser.add_argument('--mot-config-path', required=True, help='Path to the MOT model configuration file.')
    parser.add_argument('--detector-checkpoint', required=True, help='Path to the detector model checkpoint.')
    parser.add_argument('--reid-checkpoint', required=True, help='Path to the re-identification model checkpoint.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for computation.')
    parser.add_argument('--camera-id', type=int, default=0, help='ID of the webcam device.')
    return parser.parse_args()

def display_from_cam(mot_config, device='cuda:0', camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mot_model = init_model(mot_config, device=device)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        result = inference_mot(mot_model, frame, frame_id=int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        img_with_results = mot_model.show_result(frame, result, show=False, wait_time=1, out_file=None)

        cv2.imshow('Webcam - MOT Tracking', img_with_results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.mot_config_path)
    cfg.model.detector.init_cfg.checkpoint = args.detector_checkpoint
    cfg.model.reid.init_cfg.checkpoint = args.reid_checkpoint

    display_from_cam(cfg, args.device, args.camera_id)
