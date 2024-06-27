import argparse
import mmcv
from mmdet.apis import set_random_seed, train_detector as train_model
from mmdet.models import build_detector as build_model
from mmtrack.datasets import build_dataset
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with mmtracking')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--mode',help='reid or detector')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='GPU ids to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    cfg.gpu_ids = args.gpu_ids

    set_random_seed(0, deterministic=False)
    torch.cuda.empty_cache()
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    mmcv.mkdir_or_exist(cfg.work_dir)
    if args.mode == 'reid':
        model = build_model(cfg.model.reid)
    elif args.mode=='detector':
        model=build_model(cfg.model.detector)
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    train_model(model, datasets, cfg)
    print(f'Config:\n{cfg.pretty_text}')

if __name__ == '__main__':
    main()
