import argparse
import time
import datetime
import os
from pathlib import Path
from util.utils import load_json
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True, help='config path')
    parser.add_argument('--ckpt-path', type=str, default=None, help='checkpoint path to load')
    parser.add_argument('--eval', action='store_true', help='evaluate')
    parser.add_argument('--exp-name', default='base', type=str, help='experiment name')
    parser.add_argument('--seed', default=8, type=int, help='random seed')
    parser.add_argument('--top1-str', type=str, default=None, help='top1 selection strategy')
    parser.add_argument('--use-gio', action='store_true', help='use gio evaluation')
    parser.add_argument('--gio-lambdas', type=str, default="0.001,1.0,1000",
                        help='gio lambdas as start,end,num_points (linspace)')
    parser.add_argument('--mask', type=str, default="gauss")

    return parser.parse_args()


def main(kargs):
    import logging
    import numpy as np
    import random
    import torch
    from runner import Runner

    # log function
    def info(msg):
        print(msg)
        logging.info(msg)

    # set random seed
    seed = kargs.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set timer
    start_time = time.localtime()
    start_time_sec = time.time()

    # load arguments
    args = load_json(kargs.config_path)
    args['config_path'] = kargs.config_path
    if 'exp_name' not in args:
        args['exp_name'] = kargs.exp_name
    args['dataset']['frame_feat_dim'] = args['model']['frame_feat_dim']
    args['dataset']['word_feat_dim'] = args['model']['word_feat_dim']
    args['model']['max_num_words'] = args['dataset']['max_num_words']
    args['model']['top1_strategy'] = kargs.top1_str
    args['model']['use_gio'] = kargs.use_gio
    args['model']['mask_type'] = kargs.mask
    start, end, num = map(float, kargs.gio_lambdas.split(","))
    args['model']['gio_lambdas'] = np.linspace(start, end, int(num)).tolist()

    log_path = args['train']['log_path']

    # make log file
    if log_path:
        Path(log_path).mkdir(parents=True, exist_ok=True)
        time_info = time.strftime("%Y-%m-%d_%H-%M-%S.log", start_time)
        log_filename = os.path.join(log_path, "{}_{}".format(args['exp_name'], time_info))
    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    info('Starting time: %04d.%02d.%02d %02d:%02d:%02d' % (start_time.tm_year, start_time.tm_mon, start_time.tm_mday, start_time.tm_hour, start_time.tm_min, start_time.tm_sec))

    args['train']['save_path'] = os.path.join(args['train']['save_path'], "{}_{}".format(args['exp_name'], args['model']['mask_type']))

    # log arguments
    kargs_json = json.dumps(vars(kargs), indent=4)
    info(kargs_json)
    args_json = json.dumps(args, indent=4)
    info(args_json)

    # make base runner
    runner = Runner(args)

    # load checkpoint
    if kargs.ckpt_path:
        runner._load_model(kargs.ckpt_path)

    # set mode
    if kargs.eval:
        runner.eval()
        try:
            if getattr(runner, "use_wandb", False) and getattr(runner, "wandb_inited", False):
                import wandb
                wandb.finish(quiet=True)
        except Exception:
            pass
        # turn timer off
        end_time = time.localtime()
        info('Ending time: %04d.%02d.%02d %02d:%02d:%02d' % (end_time.tm_year, end_time.tm_mon, end_time.tm_mday, end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
        taken_time = str(datetime.timedelta(seconds=time.time()-start_time_sec)).split(".")
        info('Total Time taken: {}'.format(taken_time[0]))
        return
    runner.train()

    # turn timer off
    end_time = time.localtime()
    info('Ending time: %04d.%02d.%02d %02d:%02d:%02d' % (end_time.tm_year, end_time.tm_mon, end_time.tm_mday, end_time.tm_hour, end_time.tm_min, end_time.tm_sec))
    taken_time = str(datetime.timedelta(seconds=time.time()-start_time_sec)).split(".")
    info('Total Time taken: {}'.format(taken_time[0]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
