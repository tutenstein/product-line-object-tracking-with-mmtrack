import argparse
import mmcv
from mmdet.apis import set_random_seed
from mmtrack.apis import init_model, single_gpu_test
from mmtrack.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel

def parse_args():
    parser = argparse.ArgumentParser(description='Test a MOT model on a dataset.')
    parser.add_argument('--config', help='Test config file path', required=True)
    parser.add_argument('--detector-checkpoint', help='Path to the detector checkpoint', required=True)
    parser.add_argument('--reid-checkpoint', help='Path to the re-identification checkpoint', required=True)
    parser.add_argument('--work-dir', help='The dir to save logs and results', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.detector.init_cfg.checkpoint = args.detector_checkpoint
    cfg.model.reid.init_cfg.checkpoint = args.reid_checkpoint
    cfg.work_dir = args.work_dir
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.data.test.test_mode = True

    print(f'Config:\n{cfg.pretty_text}')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = init_model(cfg)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'by_epoch']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=['track']))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)

if __name__ == '__main__':
    main()
