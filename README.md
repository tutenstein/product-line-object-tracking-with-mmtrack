
# Product Line Object Tracking with mmtrack

A brief description of what this project does and who it's for


## Installation

Install cuda

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install mmtrack
```bash
!git clone https://github.com/open-mmlab/mmtracking.git
```

## Running Tests

If you want to test whether it works properly, you can download the weights here and try it.

```bash
  python tools/processVideo.py --input-dir test.mp4 --config models/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py --detector-checkpoint detector.pth --reid-checkpoint reid.pth --output-video demo.mp4
```

## Demo
You can enter the guick_run notebook and make a demo with the video called coconut. Just run it step by step and make sure that variables such as data path are correct in the config files.

