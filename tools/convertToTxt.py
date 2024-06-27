import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON annotation data to TXT format.")
    parser.add_argument('--json-file', required=True, help="Path to the input JSON file.")
    parser.add_argument('--output-file', required=True, help="Path to the output TXT file.")
    parser.add_argument('--image-width', type=int, required=True, help="Width of the images.")
    parser.add_argument('--image-height', type=int, required=True, help="Height of the images.")
    return parser.parse_args()

def round_floats(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = round_floats(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = round_floats(data[i])
    elif isinstance(data, float):
        data = round(data, 3)
    return data

def json_to_txt(json_file, output_file, image_width, image_height):
    with open(json_file, 'r') as file:
        try:
            data = json.load(file)
            data = round_floats(data)
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            return

        frame_data = []

        target_id_counter = 1
        for item in data:
            for annotation in item['annotations']:
                for result in annotation['result']:
                    sequence = result['value']['sequence']
                    for data_point in sequence:
                        frame_index = data_point.get('frame', '')
                        bbox_left = data_point.get('x', '') / 100 * image_width
                        bbox_top = data_point.get('y', '') / 100 * image_height
                        bbox_width = data_point.get('width', '') / 100 * image_width
                        bbox_height = data_point.get('height', '') / 100 * image_height
                        score = 1
                        object_category = 1
                        time = round(data_point.get('time', ''), 3)
                        target_id = target_id_counter

                        frame_data.append((frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, time))
                    target_id_counter += 1

        frame_data.sort(key=lambda x: x[0])

        with open(output_file, 'w') as f:
            f.write("frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,time\n")
            for frame in frame_data:
                f.write(f"{frame[0]},{frame[1]},{frame[2]:.3f},{frame[3]:.3f},{frame[4]:.3f},{frame[5]:.3f},{frame[6]},{frame[7]},{frame[8]:.3f}\n")
        print('successfully created output.txt')
if __name__ == '__main__':
    args = parse_args()
    json_to_txt(args.json_file, args.output_file, args.image_width, args.image_height)
