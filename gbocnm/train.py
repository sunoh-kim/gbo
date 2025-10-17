import argparse
import time
import datetime
import os
from pathlib import Path

from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='config file path')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    parser.add_argument('--tag', default='base', type=str, help='experiment tag')
    parser.add_argument('--seed', default=8, type=int, help='random seed')
    parser.add_argument('--exp-name', default='base', type=str, help='experiment name')
    parser.add_argument('--use-gio', action='store_true', help='use GIO evaluation')
    parser.add_argument('--gio-lambdas', type=str, default="0.001,1.0,1000",
                        help='gio lambdas as start,end,num_points (linspace)')
    return parser.parse_args()


def main(kargs):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner
    # logger helper
    def info(msg):
        print(msg)
        logging.info(msg)
    seed = kargs.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # timer
    start_time = time.localtime()
    start_time_sec = time.time()
    if kargs.log_dir:
        Path(kargs.log_dir).mkdir(parents=True, exist_ok=True)
        log_filename = time.strftime("%Y-%m-%d_%H-%M-%S.log", time.localtime())
        log_filename = os.path.join(kargs.log_dir, "{}_{}".format(kargs.tag, log_filename))
    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    info('Starting time: %04d.%02d.%02d %02d:%02d:%02d' % (start_time.tm_year, start_time.tm_mon, start_time.tm_mday, start_time.tm_hour, start_time.tm_min, start_time.tm_sec))

    args = load_json(kargs.config_path)
    args['train']['model_saved_path'] = os.path.join(args['train']['model_saved_path'], kargs.tag)

    if 'exp_name' not in args:
        args['exp_name'] = kargs.exp_name
    # GIO sweep 설정을 args['model']에 주입 (runner.eval에서 사용)
    start, end, num = map(float, kargs.gio_lambdas.split(","))
    args.setdefault('model', {})
    args['model']['use_gio'] = kargs.use_gio
    args['model']['gio_lambdas'] = np.linspace(start, end, int(num)).tolist()

    logging.info(str(args))

    runner = MainRunner(args)

    if kargs.resume is not None:
        runner._load_model(kargs.resume)
    if kargs.eval:
        runner.eval()
        try:
            if getattr(runner, "use_wandb", False) and getattr(runner, "wandb_inited", False):
                import wandb
                wandb.finish(quiet=True)
        except Exception:
            pass
        # timer off
        end_time = time.localtime()
        info('Ending time: %04d.%02d.%02d %02d:%02d:%02d' % (end_time.tm_year, end_time.tm_mon, end_time.tm_mday, end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
        taken_time = str(datetime.timedelta(seconds=time.time()-start_time_sec)).split(".")
        info('Total Time taken: {}'.format(taken_time[0]))     
        return
    runner.train()
    # timer off
    end_time = time.localtime()
    info('Ending time: %04d.%02d.%02d %02d:%02d:%02d' % (end_time.tm_year, end_time.tm_mon, end_time.tm_mday, end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
    taken_time = str(datetime.timedelta(seconds=time.time()-start_time_sec)).split(".")
    info('Total Time taken: {}'.format(taken_time[0]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
