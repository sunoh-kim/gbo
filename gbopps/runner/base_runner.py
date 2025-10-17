import collections
import logging
import os

import numpy as np
import torch

from model.loss import cal_nll_loss, main_loss_fn, sub_loss_fn
from util.utils import TimeMeter, AverageMeter, Accumulator

import random

import wandb
import math
import time

def info(msg):
    """ 
    Add log and print it
    """
    print(msg)
    logging.info(msg)


class Runner:
    """ 
    Base runner for training and evaluating
    """
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['max_epoch'] = self.args['train']['num_epochs']

        # build model
        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

        self.save_model = self.args['train']['save_model'] if 'save_model' in args['train'] else False
        self.use_wandb = self.args['train']['use_wandb'] if 'use_wandb' in args['train'] else False
        self.loss_meter = None
        self.evaluator = None
        self.counters = None
        self.wandb_inited = False

        self.use_early_end = self.args['train']['use_early_end'] if 'use_early_end' in self.args['train'] else False

        self.key_to_vis = ['R@1,mIoU', 'R@1,IoU@0.1', 'R@1,IoU@0.3', 'R@1,IoU@0.5', 'R@1,IoU@0.7', 'R@1,IoU@0.9', 
                           'R@5,mIoU', 'R@5,IoU@0.1', 'R@5,IoU@0.3', 'R@5,IoU@0.5', 'R@5,IoU@0.7', 'R@5,IoU@0.9',]

    def _wandb_init_once(self):
        if self.use_wandb and not self.wandb_inited:
            project = f"{self.args['model']['name']}_GIO_3_refact_{self.args['dataset']['name']}_Masks"
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

            run_name = f"{self.args.get('exp_name','base')}_{gio_tag}_{self.args['model']['mask_type']}"

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
                self.args['dataset']['name'],
                self.args['model']['name'],
                self.args['model']['top1_strategy']
            ]))
            self.wandb_inited = True

    def train(self):
        # make folder for save file
        if self.save_model:
            self.save_path = self.args['train']['save_path']
            os.makedirs(self.save_path, mode=0o755, exist_ok=True)
            os.system('cp %s %s'%(self.args['config_path'], os.path.join(self.save_path, 'config.json')))

        # make wandb to visualize learning curve
        if self.use_wandb:
            self._wandb_init_once()

        # start training
        for epoch in range(1, self.args['train']['num_epochs']+1):
            info('Start Epoch {}'.format(epoch))

            if self.args['train']['early_stop'] and epoch >= 15:
                break
            # start one epoch
            self._train_one_epoch(epoch)

            # make save file
            if self.save_model:
                save_path = os.path.join(self.save_path, 'model-{}.pt'.format(epoch))
                self._save_model(save_path)

            results = self.eval(epoch=epoch)

            # update wandb
            if self.use_wandb:
                if self.counters is None:
                    self._create_counters()
                self.update_counters()
                wandb_dict = {v.get_name(): v.get_average() for k, v in self.counters.items() if k in self.key_to_vis}
                wandb.log(wandb_dict, step=epoch)
                self.reset_counters()
            
            info('=' * 60)

        if self.use_wandb:
            wandb.finish()

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        # log function
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

        # start one epoch
        for bid, batch in enumerate(self.train_loader, 1):
            # forward
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            output = self.model(epoch=epoch, **net_input)

            # compute loss
            # main losses (reconstruction loss)
            loss, loss_dict = main_loss_fn(**output, num_props=self.model.num_props, mask_list=self.model.pos_mask_list, **self.args['loss'])

            # sub losses (pushing loss, pulling loss, intra-video contrastive loss)
            sub_loss, sub_loss_dict = sub_loss_fn(**output, num_props=self.model.num_props, mask_list=self.model.pos_mask_list, **self.args['loss'])
            loss_dict.update(sub_loss_dict)
            loss = loss + sub_loss

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            # log
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

        self.loss_meter = loss_meter
        
    def _log_selected_examples(self, batch_raw, durations_np, gt_np, selected_props_np, top1_idx_np,
                            target_ids={1,2,3}):
        try:
            B = len(batch_raw)
            chosen_norm = selected_props_np[np.arange(B), top1_idx_np]
            pred_secs = chosen_norm * durations_np[:, None]

            for i in range(B):
                vid = batch_raw[i][0]
                vid_int = vid

                if vid_int in target_ids:
                    sentence  = batch_raw[i][3]
                    gt_secs   = gt_np[i].tolist()
                    pred_i    = pred_secs[i].tolist()
                    info(f"[inference] vid={vid} | sentence={sentence} | gt(sec)={gt_secs} | pred(sec)={pred_i}")
        except Exception as e:
            info(f"[inference] example logging skipped due to error: {e}")

    def eval(self, save=None, epoch=0):
        # evaluate
        self.model.eval()

        if self.use_wandb:
            self._wandb_init_once()

        with torch.no_grad():
            use_gio = self.args['model'].get('use_gio', False)
            gio_lambdas = self.args['model'].get('gio_lambdas', np.linspace(0.001, 1.000, 1000).tolist())

            base_key = f"{gio_lambdas[0]:.3f}"

            metrics_loggers = {f"{lmb:.3f}": collections.defaultdict(lambda: AverageMeter())
                   for lmb in gio_lambdas}


            gpu_mem_peak_gb = None
            gpu_mem_reserved_gb = None
            cpu_rss_gb = None

            device = torch.cuda.current_device() if torch.cuda.is_available() else None
            if torch.cuda.is_available():
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

            avg_P_accum = 0
            avg_P_count = 0
            for bid, batch in enumerate(self.test_loader, 1):
                durations = np.asarray([i[1] for i in batch['raw']])
                gt = np.asarray([i[2] for i in batch['raw']])

                bsz = len(durations)
                total_samples += bsz
                if bid > warmup_batches:
                    core_start_time = time.time()

                # forward
                net_input = move_to_cuda(batch['net_input'])
                output = self.model(epoch=epoch, **net_input)
                bsz = len(durations)

                mask_list = self.model.pos_mask_list
                neg_mask_list = self.model.neg_mask_list

                # compute loss
                nll_losses = []
                for mask in mask_list:
                    words_logits = output['words_logits'][mask]
                    num_props_i = words_logits.size(0)//bsz

                    words_mask = output['words_mask'].unsqueeze(1) \
                        .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                    words_id = output['words_id'].unsqueeze(1) \
                        .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                    nll_loss, acc = cal_nll_loss(words_logits, words_id, words_mask)

                    nll_losses.append(nll_loss.view(bsz, num_props_i))

                nll_losses_sort, nll_loss_idx = torch.cat(nll_losses, 1).sort(dim=-1)
                nll_losses_sort = nll_losses_sort.detach().cpu().numpy()

                # predict temporal location
                left = torch.cat(list(output['prop_lefts'].values()), 1).gather(index=nll_loss_idx, dim=-1)
                right = torch.cat(list(output['prop_rights'].values()), 1).gather(index=nll_loss_idx, dim=-1)

                width = right - left
                

                if use_gio:

                    for gio_lambda in gio_lambdas:
                        # print(gio_lambda)
                        gio_weight = math.sqrt((-2) * math.log(gio_lambda))

                        selected_props = torch.stack([torch.clamp(left+(1/2-gio_weight)*width, min=0), 
                                                    torch.clamp(right-(1/2-gio_weight)*width, max=1)], dim=-1)

                        selected_props = selected_props.cpu().numpy()
                        gt_norm = gt / durations[:, np.newaxis]

                        num_all_props = selected_props.shape[1]
                        k = min(num_all_props, 5)

                        # top-1 selection strategy
                        if self.args['model']['top1_strategy'] == 'only_loss':
                            votes = 1 - np.divide(nll_losses_sort, np.max(nll_losses_sort, axis=1, keepdims=True))
                        else:
                            if self.args['model']['top1_strategy'] == 'only_iou':
                                c = np.ones((bsz, num_all_props))
                            elif self.args['model']['top1_strategy'] == 'iou_lossmax':
                                c = 1 - np.divide(nll_losses_sort, np.max(nll_losses_sort, axis=1, keepdims=True))
                            elif self.args['model']['top1_strategy'] == 'iou_losssum':
                                c = 1 - np.divide(nll_losses_sort, np.sum(nll_losses_sort, axis=1, keepdims=True))

                            votes = np.zeros((bsz, num_all_props))
                            for i in range(num_all_props):
                                for j in range(num_all_props):
                                    iou = calculate_IoU((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                    iou = iou * c[:, j]
                                    votes[:, i] = votes[:, i] + iou

                        idx = np.argmax(votes, axis=1)

                        # self._log_selected_examples(
                        #     batch_raw=batch['raw'],
                        #     durations_np=durations,
                        #     gt_np=gt, 
                        #     selected_props_np=selected_props,
                        #     top1_idx_np=idx,       
                        #     target_ids={"v_6ChRD-1NwSg","v_huUb8mM5fv4","v_z85nM9V4058","v_Y1UwPTU61uk"}
                        # )

                        res = top_1_metric(selected_props[np.arange(bsz), idx], gt_norm)
                        
                        lmb_key = f"{gio_lambda:.3f}"
                        # compute result of top-n
                        for key, v in res.items():
                            metrics_loggers[lmb_key]['R@1,'+key].update(v, bsz)
                        res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt_norm)
                        for key, v in res.items():
                            metrics_loggers[lmb_key][f'R@{k},'+key].update(v, bsz)
                            
                else:
                    # print(gio_lambda)

                    selected_props = torch.stack([torch.clamp(left, min=0), 
                                                torch.clamp(right, max=1)], dim=-1)

                    selected_props = selected_props.cpu().numpy()
                    gt_norm = gt / durations[:, np.newaxis]

                    num_all_props = selected_props.shape[1]
                    k = min(num_all_props, 5)

                    # top-1 selection strategy
                    if self.args['model']['top1_strategy'] == 'only_loss':
                        votes = 1 - np.divide(nll_losses_sort, np.max(nll_losses_sort, axis=1, keepdims=True))
                    else:
                        if self.args['model']['top1_strategy'] == 'only_iou':
                            c = np.ones((bsz, num_all_props))
                        elif self.args['model']['top1_strategy'] == 'iou_lossmax':
                            c = 1 - np.divide(nll_losses_sort, np.max(nll_losses_sort, axis=1, keepdims=True))
                        elif self.args['model']['top1_strategy'] == 'iou_losssum':
                            c = 1 - np.divide(nll_losses_sort, np.sum(nll_losses_sort, axis=1, keepdims=True))

                        votes = np.zeros((bsz, num_all_props))
                        for i in range(num_all_props):
                            for j in range(num_all_props):
                                iou = calculate_IoU((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                iou = iou * c[:, j]
                                votes[:, i] = votes[:, i] + iou

                    idx = np.argmax(votes, axis=1)

                    # self._log_selected_examples(
                    #     batch_raw=batch['raw'],
                    #     durations_np=durations,
                    #     gt_np=gt, 
                    #     selected_props_np=selected_props,
                    #     top1_idx_np=idx,       
                    #     target_ids={"v_6ChRD-1NwSg","v_huUb8mM5fv4","v_z85nM9V4058","v_Y1UwPTU61uk"}
                    # )
                    res = top_1_metric(selected_props[np.arange(bsz), idx], gt_norm)

                    # compute result of top-n
                    for key, v in res.items():
                        metrics_loggers[base_key]['R@1,'+key].update(v, bsz)
                    res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt_norm)
                    for key, v in res.items():
                        metrics_loggers[base_key]['R@%d,'%(k)+key].update(v, bsz)

                if bid > warmup_batches:
                    core_time += (time.time() - core_start_time)
                    measured_samples += bsz
                avg_P_accum += num_all_props
                avg_P_count += 1

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

            avg_P = (avg_P_accum / avg_P_count) if avg_P_count > 0 else None
            L = len(gio_lambdas) if use_gio else 1
            if avg_P is not None:
                if use_gio:
                    info(f'Complexity (approx): O(L·P^2 + P log P), with L={L}, ~P={avg_P:.1f}')
                else:
                    info(f'Complexity (approx): O(P^2 + P log P), ~P={avg_P:.1f}')

            if use_gio:
                if self.use_wandb:
                    ds = self.args['dataset']['name']
                    top1 = self.args['model']['top1_strategy']
                    prefix = f"{ds}/{top1}/"

                    cols = ['dataset', 'top1', 'gio_lambda'] + self.key_to_vis
                    table = wandb.Table(columns=cols)

                    timing_payload = {"timing/eval_total_wall_s": time_taken}
                    if measured_samples > 0:
                        timing_payload.update({
                            "timing/ms_per_sample_core": core_time / measured_samples * 1000.0
                        })

                    if gpu_mem_peak_gb is not None:
                        timing_payload["memory/gpu_peak_gb"] = gpu_mem_peak_gb
                    if gpu_mem_reserved_gb is not None:
                        timing_payload["memory/gpu_reserved_gb"] = gpu_mem_reserved_gb
                    if hasattr(self, "total_params"):
                        timing_payload["model/params_total_m"] = self.total_params / 1e6
                        timing_payload["model/params_trainable_m"] = self.trainable_params / 1e6
                    if avg_P is not None:
                        timing_payload["complexity/avg_P"] = avg_P
                        timing_payload["complexity/L"] = L

                    wandb.log(timing_payload, step=0)

                    for lmb in gio_lambdas:
                        lmb_key = f"{lmb:.3f}"
                        logger = metrics_loggers[lmb_key]
                        row = [ds, top1, float(lmb)]
                        log_payload = {}
                        for k in self.key_to_vis:
                            if k in logger:
                                v = logger[k].avg
                                log_payload[prefix + k] = v
                                row.append(v)
                            else:
                                row.append(float('nan'))
                        wandb.log(log_payload, step=int(round(lmb*1000)))
                        table.add_data(*row)

                    wandb.log({f"{prefix}eval_table": table})


                best_vals = {}
                for lmb in gio_lambdas:
                    lmb_key = f"{lmb:.3f}"
                    logger = metrics_loggers[lmb_key]
                    vals = {k: logger[k].avg for k in self.key_to_vis if k in logger}
                    for k, v in vals.items():
                        if k not in best_vals or v > best_vals[k][0]:
                            best_vals[k] = (v, lmb)

                info("=== Best results across λ ===")
                for k in sorted(best_vals.keys()):
                    val, lmb = best_vals[k]
                    info(f"{k}: {val:.4f} at λ={lmb:.3f}")

                best_metric, best_lambda = best_vals['R@1,mIoU']
                best_key = f"{best_lambda:.3f}"
                self.evaluator = metrics_loggers[best_key]

                info(f"Evaluator set to best λ={best_lambda:.3f} (R@1,mIoU={best_metric:.4f})")
                info('=' * 60)

                return metrics_loggers[best_key]
            else:
        
                self.evaluator = metrics_loggers[base_key]

                info("=== Results (no GIO) ===")
                for k in self.key_to_vis:
                    if k in metrics_loggers[base_key]:
                        info(f"{k}: {metrics_loggers[base_key][k].avg:.4f}")

                info('=' * 60)

                # update wandb
                if self.use_wandb:
                    ds = self.args['dataset']['name']
                    top1 = self.args['model']['top1_strategy']
                    prefix = f"{ds}/{top1}/"

                    cols = ['dataset', 'top1', 'gio_lambda'] + self.key_to_vis
                    table = wandb.Table(columns=cols)

                    timing_payload = {
                        "timing/eval_total_wall_s": time_taken
                    }
                    if measured_samples > 0:
                        timing_payload.update({
                            "timing/ms_per_sample_core": core_time / measured_samples * 1000.0
                        })
                    if gpu_mem_peak_gb is not None:
                        timing_payload["memory/gpu_peak_gb"] = gpu_mem_peak_gb
                    if gpu_mem_reserved_gb is not None:
                        timing_payload["memory/gpu_reserved_gb"] = gpu_mem_reserved_gb
                    if hasattr(self, "total_params"):
                        timing_payload["model/params_total_m"] = self.total_params / 1e6
                        timing_payload["model/params_trainable_m"] = self.trainable_params / 1e6
                    if avg_P is not None:
                        timing_payload["complexity/avg_P"] = avg_P
                        timing_payload["complexity/L"] = L

                    wandb.log(timing_payload, step=0)

                    logger = metrics_loggers[base_key]
                    row = [ds, top1, float(gio_lambdas[0])]
                    log_payload = {}
                    for k in self.key_to_vis:
                        if k in logger:
                            v = logger[k].avg
                            log_payload[prefix + k] = v
                            row.append(v)
                        else:
                            row.append(float('nan'))
                    wandb.log(log_payload, step=0)
                    table.add_data(*row)

                    wandb.log({f"{prefix}eval_table": table})

                self.evaluator = metrics_loggers[base_key]

                info('=' * 60)

                return metrics_loggers[base_key]

    def _build_dataset(self):
        import dataset as da
        import pickle
        from torch.utils.data import DataLoader

        args = self.args['dataset']
        cls = getattr(da, args['name'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        # self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):

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

    def _build_model(self):
        import model

        model_config = self.args['model']
        self.model = getattr(model, model_config['name'], None)(model_config)
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)
        
        self.total_params = total_num
        self.trainable_params = trainable_num

    def _build_optimizer(self):
        from model.optimizer import AdamOptimizer
        from model.optimizer.lr_scheduler import InverseSquareRootSchedule

        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["optimizer"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

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
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        info('load model from {}, num_updates {}.'.format(path, self.num_updates))

    def _create_counters(self):
        self.counters = dict()
        if self.loss_meter is not None:
            for k, _ in self.loss_meter.items():
                self.counters[k] = Accumulator(k)
        if self.evaluator is not None:
            for k, _ in self.evaluator.items():
                self.counters[k] = Accumulator(k)
    
    def update_counters(self):
        if self.loss_meter is not None:
            for k, v in self.loss_meter.items():
                self.counters[k].add(v.sum_all, v.count_all)
        if self.evaluator is not None:
            for k, v in self.evaluator.items():
                self.counters[k].add(v.sum_all, v.count_all)
    
    def reset_counters(self):
        for k, v in self.counters.items():
            v.reset()


def calculate_IoU(i0, i1):
    """ 
    compute IoU
    """
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def top_n_metric(preds, label):
    """ 
    compute result of top-n
    """
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    """ 
    compute result of top-1
    """
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    """ 
    apply to sample (dict(), list(), etc...)
    """
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


def move_to_cuda(sample):
    """ 
    move to cuda
    """
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
