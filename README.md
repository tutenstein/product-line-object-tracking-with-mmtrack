
# Product Line Object Tracking with mmtrack

This project utilizes mmtrack for efficient object tracking in product line environments.



## Installation
To get started, follow these steps:

1. Install cuda
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
2. Install mmtrack
```bash
git clone https://github.com/open-mmlab/mmtracking.git
```
 
## Running Tests

To verify the functionality, you can download the necessary weights[here].(https://drive.google.com/file/d/17USVkFwbzz2ZuDlSV2iuOwYC0Ypaqcfa/view?usp=sharing) 

```bash
  python tools/processVideo.py --input-dir test.mp4 --config models/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py --detector-checkpoint detector.pth --reid-checkpoint reid.pth --output-video demo.mp4
```

## Demo
Explore the quick_run notebook to execute a demo with the 'coconut' video. Ensure all variables, such as data paths in the configuration files, are correctly set.



![me](https://github.com/tutenstein/product-line-object-tracking-by-mmtrack/blob/main/result.gif)

## Test Results for Coconut Tracking and Counting in Product Line
The following table summarizes the performance metrics obtained from tracking and counting coconuts in a product line using advanced computer vision techniques:

### Tracking Performance

| Metric       | Value   |
|--------------|---------|
| HOTA         | 84.305  |
| DetA         | 82.454  |
| AssA         | 86.44   |
| DetRe        | 87.39   |
| DetPr        | 86.593  |
| AssRe        | 91.103  |
| AssPr        | 88.949  |
| LocA         | 88.359  |
| OWTA         | 86.905  |
| HOTA(0)      | 95.897  |
| LocA(0)      | 87.25   |
| HOTALocA(0)  | 83.67   |

### Counting Metrics

| Metric       | Value   |
|--------------|---------|
| Detections   | 2633    |
| Ground Truth Detections | 2609 |
| Unique IDs   | 36      |
| Ground Truth IDs | 36   |

### Overall Metrics

| Metric       | Value   |
|--------------|---------|
| IDF1         | 97.5%   |
| IDP          | 97.0%   |
| IDR          | 97.9%   |
| Recall (Rcll)| 98.0%   |
| Precision (Prcn)| 97.1% |
| GT           | 36      |
| MT (Mostly Tracked) | 35 |
| PT (Partially Tracked) | 0 |
| ML (Mostly Lost) | 1    |
| False Positives (FP)| 77 |
| False Negatives (FN)| 53 |
| ID Switches (IDs) | 1   |
| Fragmentation (FM)| 1   |
| MOTA         | 95.0%   |
| MOTP         | 0.128   |
| ID Switches (IDt)| 0    |
| ID Ass. Errors (IDa)| 1 |
| ID Merge Errors (IDm)| 0|
| HOTA         | 84.305  |

These results demonstrate the accuracy and robustness of the tracking and counting system in a real-world product line scenario, highlighting its effectiveness in identifying and monitoring coconuts throughout the production process.
