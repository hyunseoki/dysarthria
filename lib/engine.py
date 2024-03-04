import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BaseTrainer:
    def __init__(self, model, train_loader, valid_loader, num_epochs, save_path, tokenizer, metric_func, optimizer, device, weight, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metric_func = metric_func
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.weight = weight

        self.scheduler = scheduler
        self.num_epochs = num_epochs

        self.elapsed_time = None

        self.loss = {'train': self._init_loss(list), 'valid': self._init_loss(list)}
        self.metric = {'train': self._init_metric(list), 'valid': self._init_metric(list)}
        self.lr_curve = list()


    def _init_loss(self, dtype):
        return {'ctc_loss': dtype(), 'clf_loss': dtype()}


    def _init_metric(self, dtype):
        return {'clf_acc': dtype(), 'cer': dtype(), 'wer': dtype()}


    def fit(self):
        if self.device == 'cpu':
            print('[info msg] Start training the model on CPU')
            self.model.to(self.device)
        else:
            print(f'[info msg] Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
            self.model.to(self.device)

        best_metric = -float('inf')
        startTime = datetime.now()

        print('=' * 50)
        print('[info msg] training start !!')
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))

            for phase in ['Train', 'Valid']:
                log = self._one_epoch(phase=phase)

                for loss_key in ['ctc_loss', 'clf_loss']:
                    self.loss[phase.lower()][loss_key].append(log[loss_key])
                for metric_key in ['clf_acc', 'wer', 'cer']:
                    self.metric[phase.lower()][metric_key].append(log[metric_key])

            cur_acc = self.metric['valid']['clf_acc'][-1]
            if cur_acc > best_metric:
                best_metric = cur_acc
                self._save_model(param=self.model.state_dict(), fn='model_best.pth')

            self.lr_curve.append(self.optimizer.param_groups[0]['lr'])
            if self.scheduler is not None:
                self.scheduler.step()

        self.elapsed_time = datetime.now() - startTime
        self._save_result()


    def _one_epoch(self, phase):
        running_loss = self._init_loss(AverageMeter)
        running_metric = self._init_metric(AverageMeter)

        if phase.lower() == 'train':
            self.model.train()
            loop = tqdm.tqdm(self.train_loader, total=len(self.train_loader), desc=phase)
        else:
            self.model.eval()
            loop = tqdm.tqdm(self.valid_loader, total=len(self.valid_loader), desc=phase)
        
        for sample in loop:
            if phase.lower() == 'train':
                ret = self._step(sample)
                loss = (1-self.weight) * ret['clf_loss'] + self.weight * ret['ctc_loss']
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    ret = self._step(sample)
    
            clf_acc = self.metric_func['cls_metric'](
                real=sample['clf_labels'].detach().cpu().numpy().tolist(),
                pred=ret['clf_logits'].argmax(1).detach().cpu().numpy().tolist(),
            )
            wer, cer = self.metric_func['ctc_metric'](
                true=sample['ctc_labels'].detach().cpu().numpy(), 
                pred=ret['ctc_logits'].detach().cpu().numpy(), 
                tokenizer=self.tokenizer
            )

            running_loss['clf_loss'].update(ret['clf_loss'].item(), sample['input_values'].shape[0])
            running_loss['ctc_loss'].update(ret['ctc_loss'].item(), sample['input_values'].shape[0])
            running_metric['clf_acc'].update(clf_acc, sample['input_values'].shape[0])
            running_metric['wer'].update(wer, sample['input_values'].shape[0])
            running_metric['cer'].update(cer, sample['input_values'].shape[0])

            log = {
                'clf_loss': running_loss['clf_loss'].avg,
                'clf_acc': running_metric['clf_acc'].avg, 
                'ctc_loss': running_loss['ctc_loss'].avg, 
                'wer': running_metric['wer'].avg,
                'cer': running_metric['cer'].avg,
            }
            loop.set_postfix({k: f"{v:.3f}" for k, v in log.items()})

        return log


    @torch.cuda.amp.autocast()
    def _step(self, sample):
        sample = {k: v.to(self.device) for k, v in sample.items()}
        ret = self.model(**sample)

        return ret


    def _save_model(self, param, fn):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        full_name = os.path.join(self.save_path, fn)
        torch.save(param, full_name)
        print('MODEL IS SAVED TO {}!!!'.format(full_name))


    def _save_result(self):
        curves_dict = {}

        for phase, phase_loss in self.loss.items():
            for loss_key, loss_values in phase_loss.items():
                curves_dict[f'{phase}_{loss_key}'] = loss_values

        for phase, phase_metric in self.metric.items():
            for metric_key, metric_values in phase_metric.items():
                curves_dict[f'{phase}_{metric_key}'] = metric_values

        df_learning_curves = pd.DataFrame.from_dict(curves_dict)
        df_learning_curves.to_csv(os.path.join(self.save_path, 'learning_curves.csv'), sep=',')

        best_train_row = df_learning_curves.loc[df_learning_curves['train_clf_acc'].idxmax()]
        best_valid_row = df_learning_curves.loc[df_learning_curves['valid_clf_acc'].idxmax()]

        print('=' * 50)
        print('[info msg] training is done')
        print(f"Time taken: {self.elapsed_time}")
        print(f"best clf metric is {best_valid_row['valid_clf_acc']} w/ clf loss {best_valid_row['valid_clf_loss']}, cer {best_valid_row['valid_cer']}, wer {best_valid_row['valid_wer']} at epoch {best_valid_row.name}")

        print('=' * 50)
        print('[info msg] model weight and log is save to {}'.format(self.save_path))

        with open(os.path.join(self.save_path, 'log.txt'), 'w') as f:
            f.write(f'total ecpochs : {self.num_epochs}\n')
            f.write(f'time taken : {self.elapsed_time}\n')
            f.write(f'best train clf metric is {best_train_row["train_clf_acc"]} w/ clf loss {best_train_row["train_clf_loss"]}, cer {best_train_row["train_cer"]}, wer {best_train_row["train_wer"]} at epoch {best_train_row.name}\n')
            f.write(f'best valid clf metric is {best_valid_row["valid_clf_acc"]} w/ clf loss {best_valid_row["valid_clf_loss"]}, cer {best_valid_row["valid_cer"]}, wer {best_valid_row["valid_wer"]} at epoch {best_valid_row.name}\n')

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(df_learning_curves['train_clf_loss'], label='train_clf_loss')
        plt.plot(df_learning_curves['valid_clf_loss'], label='valid_clf_loss')
        plt.axvline(x=best_valid_row.name, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.title('CLF_Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('CE loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(df_learning_curves['train_clf_acc'], label='train_clf_acc')
        plt.plot(df_learning_curves['valid_clf_acc'], label='valid_clf_acc')
        plt.axvline(x=best_valid_row.name, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.title('CLF_Acc Curve')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'clf_curve.png'))
        plt.close()

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(df_learning_curves['train_ctc_loss'], label='train_ctc_loss')
        plt.plot(df_learning_curves['valid_ctc_loss'], label='valid_ctc_loss')
        plt.title('CTC_Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('CTC Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(df_learning_curves['train_cer'], label='train_cer')
        plt.plot(df_learning_curves['valid_cer'], label='valid_cer')
        plt.title('CER Curve')
        plt.xlabel('Epoch')
        plt.ylabel('CER')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(df_learning_curves['train_wer'], label='train_wer')
        plt.plot(df_learning_curves['valid_wer'], label='valid_wer')
        plt.title('WER Curve')
        plt.xlabel('Epoch')
        plt.ylabel('WER')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'sr_curve.png'))
        plt.close()

        plt.figure(figsize=(15,5))
        plt.title('lr_rate curve')
        plt.grid(True)
        plt.plot(self.lr_curve)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'lr_history.png'))