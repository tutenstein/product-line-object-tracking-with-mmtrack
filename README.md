
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

