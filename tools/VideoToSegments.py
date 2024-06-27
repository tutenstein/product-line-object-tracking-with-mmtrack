import os
import cv2
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Segment a video and annotations into smaller clips.")
    parser.add_argument('--txt-file', required=True, help="Path to the input txt file containing annotations.")
    parser.add_argument('--video-file', required=True, help="Path to the input video file.")
    parser.add_argument('--output-dir', required=True, help="Directory to store the output segments.")
    parser.add_argument('--segment-duration', type=int, default=10, help="Duration of each segment in seconds.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second of the video.")
    return parser.parse_args()

def create_segments(txt_file, video_file, output_dir, segment_duration=10, fps=30):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps

    df = pd.read_csv(txt_file)

    segment_frames = segment_duration * fps
    num_segments = int(total_duration / segment_duration)

    for segment_index in range(num_segments):
        start_time = segment_index * segment_duration
        end_time = start_time + segment_duration
        segment_data = df[(df['time'] >= start_time) & (df['time'] < end_time)].copy()

        segment_data['frame_index'] = segment_data['frame_index'] - segment_data['frame_index'].min() + 1

        segment_output_dir = os.path.join(output_dir, f"segment_{segment_index+1}")
        img_output_dir = os.path.join(segment_output_dir, "img")
        segment_gt_dir = os.path.join(segment_output_dir, 'gt')
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(segment_gt_dir, exist_ok=True)

        segment_txt_path = os.path.join(segment_gt_dir, "gt.txt")
        segment_data.to_csv(segment_txt_path, index=False, header=False)

        seqinfo_path = os.path.join(segment_output_dir, "seqinfo.ini")
        with open(seqinfo_path, "w") as seqinfo_file:
            seqinfo_file.write("[Sequence]\n")
            seqinfo_file.write(f"name=segment_{segment_index+1}\n")
            seqinfo_file.write("imDir=img\n")
            seqinfo_file.write(f"frameRate={fps}\n")
            seqinfo_file.write(f"seqLength={segment_frames}\n")
            seqinfo_file.write("imWidth=1280\n")
            seqinfo_file.write("imHeight=720\n")
            seqinfo_file.write("imExt=.jpg\n")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
        for frame_index in range(1, segment_frames + 1):
            ret, frame = cap.read()
            if not ret:
                break

            output_frame_name = os.path.join(img_output_dir, f"{frame_index:06d}.jpg")
            cv2.imwrite(output_frame_name, frame)

    cap.release()
    print("Processing complete.")

if __name__ == '__main__':
    args = parse_args()
    create_segments(args.txt_file, args.video_file, args.output_dir, args.segment_duration, args.fps)
