import collections
import logging
import os
import time
import math

import numpy as np
import torch
from torch.utils import data

from models.loss import ivc_loss, rec_loss
from utils import TimeMeter, AverageMeter

import pickle
import copy
from pathlib import Path

import wandb
import math

def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['config']['max_num_words'] = self.args['dataset']['max_num_words']
        self.args['model']['config']['frames_input_size'] = self.args['dataset']['frame_dim']
        self.args['model']['config']['words_input_size'] = self.args['dataset']['word_dim']
        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0
        self.use_wandb = self.args['train'].get('use_wandb', True)
        self.wandb_inited = False
        self.key_to_vis = ['IoU@0.1', 'IoU@0.3', 'IoU@0.5', 'IoU@0.7', 'IoU@0.9', 'mIoU']


    def _wandb_init_once(self):
        if self.use_wandb and not self.wandb_inited:
            project = f"{self.args['model']['name']}_GIO_3_{self.args['dataset']['dataset']}"
            run_id = None
            wandb.login()

            def summarize_lambdas(lambdas, max_preview=3):
                try:
                    vals = list(map(float, lambdas))
                except Exception:
                    return "lambda=custom"
                if len(vals) == 0:
                    return "lambda=none"
                uniq = sorted(set([round(v, 6) for v in vals]))
                if len(uniq) == 1:
                    return f"lambda={uniq[0]:.3f}"
                mn, mx, n = min(vals), max(vals), len(vals)
                preview = ",".join(f"{v:.3f}" for v in vals[:max_preview])
                suffix = f"{preview}{',…' if n > max_preview else ''}"
                short = f"lambda=[{mn:.3f}..{mx:.3f}]@{n}"
                return f"{short}|{suffix}"
            
            use_gio = self.args['model'].get('use_gio', False)
            gio_tag = "gio" if use_gio else "no-gio"
            if use_gio:
                lambdas = self.args['model'].get('gio_lambdas', [])
                lambdas_str_short = summarize_lambdas(lambdas)
                gio_tag += f"-{lambdas_str_short}"

            run_name = f"{self.args.get('exp_name','base')}_{gio_tag}"

            wandb.init(
                project=project,
                name=run_name,
                config=self.args,
                dir=self.args['train'].get('wandb_path', './wandb'),
                id=run_id,
                resume="allow" if run_id else None,
                reinit=False,
                settings=wandb.Settings(start_method="thread")
            )
            wandb.run.tags = list(set(list(wandb.run.tags) + [
                self.args['dataset']['dataset'],
                self.args['model']['name']
            ]))
            self.wandb_inited = True
    def train(self):
        best_results = None
        if self.use_wandb:
            self._wandb_init_once()

        for epoch in range(self.args['train']['max_num_epochs']):
            info('Start Epoch {}'.format(epoch))
            self.model_saved_path = self.args['train']['model_saved_path']
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))

            self._train_one_epoch(epoch)
            self._save_model(save_path)
            results = self.eval()
            if best_results is None or results['mIoU'].avg > best_results['mIoU'].avg:
                best_results = results
                os.system('cp %s %s'%(save_path, os.path.join(self.model_saved_path, 'model-best.pt')))
                info('Best results have been updated.')
            info('=' * 60)
        
        if self.use_wandb:
            wandb.finish(quiet=True)
        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in best_results.items()])
        info('Best results:')
        info('|'+msg+'|')


    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            self.model.froze_mask_generator()
            self.rec_optimizer.zero_grad()
            net_input = move_to_device(batch['net_input'])
            output = self.model(**net_input)
            loss, loss_dict = rec_loss(**output, **self.args['loss'])
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.rec_optimizer.step()

            self.model.froze_reconstructor()
            self.mask_optimizer.zero_grad()
            output = self.model(**net_input)
            loss, ivc_loss_dict = ivc_loss(**output, **self.args['loss'])
            loss_dict.update(ivc_loss_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.mask_optimizer.step()

            self.num_updates += 1
            curr_lr = self.rec_lr_scheduler.step_update(self.num_updates)
            self.mask_lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

    def eval(self, data_loader=None):
        self.model.eval()
        if data_loader is None:
            data_loader = self.test_loader
        if self.use_wandb:
            self._wandb_init_once()

        with torch.no_grad():
            use_gio = bool(self.args.get('model', {}).get('use_gio', True))
            gio_lambdas = self.args.get('model', {}).get('gio_lambdas', list(np.linspace(0.001, 1.0, 1000)))
            base_key = f"{gio_lambdas[0]:.3f}"

            metrics_loggers = {f"{float(lmb):.3f}": collections.defaultdict(lambda: AverageMeter())
                               for lmb in gio_lambdas}

            gpu_mem_peak_gb = None
            gpu_mem_reserved_gb = None
            cpu_rss_gb = None
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
            try:
                import psutil, os as _os
                _proc = psutil.Process(_os.getpid())
                cpu_rss_gb = _proc.memory_info().rss / (1024**3)
            except Exception:
                cpu_rss_gb = None

            start_time = time.time()
            warmup_batches = 2
            total_samples = 0
            measured_samples = 0
            core_time = 0.0

            # metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():
                for bid, batch in enumerate(data_loader, 1):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])
                    bsz = len(durations)
                    total_samples += bsz
                    if bid > warmup_batches:
                        core_start_time = time.time()

                    net_input = move_to_device(batch['net_input'])
                    output = self.model(**net_input)
                    width = output['width'].view(bsz)
                    center = output['center'].view(bsz)

                    if use_gio:
                        for lmb in gio_lambdas:
                            if lmb <= 0:
                                continue
                            gio_weight = math.sqrt((-2) * math.log(lmb))
                            selected_props = torch.stack([
                                torch.clamp(center - gio_weight * width, min=0),
                                torch.clamp(center + gio_weight * width, max=1)
                            ], dim=-1)  # [B,2]
                            selected_props = selected_props.cpu().numpy()
                            gt_norm = gt / durations[:, np.newaxis]
                            res = top_1_metric(selected_props, gt_norm)
                            lmb_key = f"{float(lmb):.3f}"
                            for k, v in res.items():
                                metrics_loggers[lmb_key][k].update(v, bsz)
                    else:
                        selected_props = torch.stack([
                            torch.clamp(center - 0.5 * width, min=0),
                            torch.clamp(center + 0.5 * width, max=1)
                        ], dim=-1)
                        selected_props = selected_props.cpu().numpy()
                        gt_norm = gt / durations[:, np.newaxis]
                        res = top_1_metric(selected_props, gt_norm)
                        for k, v in res.items():
                            metrics_loggers[base_key][k].update(v, bsz)

                    if bid > warmup_batches:
                        core_time += (time.time() - core_start_time)
                        measured_samples += bsz


            time_taken = time.time() - start_time
            if torch.cuda.is_available():
                gpu_mem_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                try:
                    gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
                except Exception:
                    gpu_mem_reserved_gb = None
            if gpu_mem_peak_gb is not None:
                info(f'GPU Peak Memory: {gpu_mem_peak_gb:.3f} GB')
            if gpu_mem_reserved_gb is not None:
                info(f'GPU Max Reserved: {gpu_mem_reserved_gb:.3f} GB')
            if cpu_rss_gb is not None:
                info(f'CPU RSS: {cpu_rss_gb:.3f} GB')
            if hasattr(self, "total_params"):
                info(f'Params: total={self.total_params/1e6:.2f}M, trainable={self.trainable_params/1e6:.2f}M')
            info(f'Evaluation Time: {time_taken:.3f}s')
            if measured_samples > 0:
                info(f'Per Sample Time: {core_time/measured_samples*1000:.2f} ms')

            # wandb
            if self.use_wandb:
                ds = self.args['dataset']['dataset']
                prefix = f"{ds}/"
                cols = ['dataset', 'gio_lambda'] + self.key_to_vis
                table = wandb.Table(columns=cols)

                timing_payload = {"timing/eval_total_wall_s": time_taken}
                if measured_samples > 0:
                    timing_payload["timing/ms_per_sample_core"] = core_time / measured_samples * 1000.0
                if gpu_mem_peak_gb is not None:
                    timing_payload["memory/gpu_peak_gb"] = gpu_mem_peak_gb
                if gpu_mem_reserved_gb is not None:
                    timing_payload["memory/gpu_reserved_gb"] = gpu_mem_reserved_gb
                if hasattr(self, "total_params"):
                    timing_payload["model/params_total_m"] = self.total_params / 1e6
                    timing_payload["model/params_trainable_m"] = self.trainable_params / 1e6
                wandb.log(timing_payload, step=0)

                if use_gio:
                    for lmb in gio_lambdas:
                        lmb_key = f"{float(lmb):.3f}"
                        logger = metrics_loggers[lmb_key]
                        row = [ds, float(lmb)]
                        payload = {}
                        for k in self.key_to_vis:
                            v = logger[k].avg if k in logger else float('nan')
                            payload[prefix + k] = v
                            row.append(v)
                        wandb.log(payload, step=int(round(float(lmb) * 1000)))
                        table.add_data(*row)
                else:
                    logger = metrics_loggers[base_key]
                    row = [ds, float(gio_lambdas[0])]
                    payload = {}
                    for k in self.key_to_vis:
                        v = logger[k].avg if k in logger else float('nan')
                        payload[prefix + k] = v
                        row.append(v)
                    wandb.log(payload, step=int(round(float(gio_lambdas[0]) * 1000)))
                    table.add_data(*row)

                wandb.log({f"{prefix}eval_table": table})

            if use_gio:
                best_key = None
                best_val = -1
                for lmb in gio_lambdas:
                    key = f"{float(lmb):.3f}"
                    val = metrics_loggers[key]['mIoU'].avg if 'mIoU' in metrics_loggers[key] else -1
                    if val > best_val:
                        best_val = val
                        best_key = key
                info("=== Best λ by mIoU ===")
                info(f"best λ={best_key}, mIoU={best_val:.4f}")
                return metrics_loggers[best_key]
            else:
                return metrics_loggers[base_key]

    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)

        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True)
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args)
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args) if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=2,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=0)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=1) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        import models

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)
        self.total_params = total_num
        self.trainable_params = trainable_num

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule

        self.model.froze_mask_generator()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["reconstructor"]
        self.rec_optimizer = AdamOptimizer(args, parameters)
        self.rec_lr_scheduler = InverseSquareRootSchedule(args, self.rec_optimizer)

        self.model.froze_reconstructor()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["generator"]
        self.mask_optimizer = AdamOptimizer(args, parameters)
        self.mask_lr_scheduler = InverseSquareRootSchedule(args, self.mask_optimizer)
        

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.mask_lr_scheduler.step_update(self.num_updates)
        self.rec_lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_device(sample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _move(tensor):
        return tensor.to(device) if torch.is_tensor(tensor) else tensor
    return apply_to_sample(_move, sample)
