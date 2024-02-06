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
    def __init__(self, model, train_loader, valid_loader, num_epochs, save_dir, loss_func, metric_func, optimizer, device, parallel=False, mode='max', scheduler=None):
        assert mode in ['min', 'max']

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.optimizer = optimizer
        self.device = device
        self.mode = mode
        self.save_path = save_dir

        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.parallel = parallel

        self.elapsed_time = None

        self.train_loss = list()
        self.train_metric = list()

        self.valid_loss = list()
        self.valid_metric = list()

        self.lr_curve = list()


    def fit(self):
        if self.device == 'cpu':
            print('[info msg] Start training the model on CPU')
            self.model.to(self.device)

        elif self.parallel and torch.cuda.device_count() > 1:
            print(f'Start training the model on {torch.cuda.device_count()} '
                  f'{torch.cuda.get_device_name(torch.cuda.current_device())} in parallel')
            self.model = torch.nn.DataParallel(self.model)

        else:
            print(f'[info msg] Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
            self.model.to(self.device)

        print('=' * 50)

        if self.mode =='max':
            best_metric = -float('inf')
        else:
            best_metric = float('inf')
        startTime = datetime.now()

        print('=' * 50)
        print('[info msg] training start !!')
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            train_epoch_loss, train_epoch_metric = self.one_epoch(phase='Train')
            self.train_loss.append(train_epoch_loss)
            self.train_metric.append(train_epoch_metric)

            valid_epoch_loss, valid_epoch_metric = self.one_epoch(phase='Valid')
            self.valid_loss.append(valid_epoch_loss)
            self.valid_metric.append(valid_epoch_metric)
            self.lr_curve.append(self.optimizer.param_groups[0]['lr'])

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_epoch_metric)
                else:
                    self.scheduler.step()

            if (self.mode =='min' and valid_epoch_metric < best_metric) or \
               (self.mode =='max' and valid_epoch_metric > best_metric) :
                best_metric = valid_epoch_metric
                self.save_model(param=self.model.state_dict(), fn='model_best.pth')

        self.elapsed_time = datetime.now() - startTime
        self.save_result()


    def save_model(self, param, fn):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        full_name = os.path.join(self.save_path, fn)
        torch.save(param, full_name)
        print('MODEL IS SAVED TO {}!!!'.format(full_name))


    def step(self, sample):
        img, csv, label = sample['image'].to(self.device), sample['csv'].to(self.device), sample['label'].to(self.device)
        logit = self.model(img, csv)

        loss = self.loss_func(logit, label)
        metric = self.metric_func(label.detach().cpu().numpy().tolist(), logit.argmax(1).detach().cpu().numpy().tolist())
        return loss, metric


    def one_epoch(self, phase):
        running_metric = AverageMeter()
        running_loss = AverageMeter()

        if phase.lower()=='train':
            self.model.train()
            data_loader = self.train_loader
            scaler = torch.cuda.amp.GradScaler()
        else:
            self.model.eval()
            data_loader = self.valid_loader

        torch.cuda.empty_cache()
        with tqdm.tqdm(data_loader, total=len(data_loader), desc=phase, file=sys.stdout) as iterator:
            for sample in iterator:
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(phase.lower()=='train'):
                        loss, metric = self.step(sample)

                if phase.lower()=='train':
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                running_loss.update(loss.item(), sample['input_values'].shape[0])
                running_metric.update(metric.item(), sample['input_values'].shape[0])
                log = 'loss - {:.5f}, metric - {:.5f}'.format(running_loss.avg, running_metric.avg)
                iterator.set_postfix_str(log)

        return running_loss.avg, running_metric.avg


    def save_result(self):
        train_loss = np.array(self.train_loss)
        train_metric = np.array(self.train_metric)
        valid_loss = np.array(self.valid_loss)
        valid_metric = np.array(self.valid_metric)

        if self.mode =='max':
            best_train_metric_pos = np.argmax(train_metric)
            best_val_metric_pos = np.argmax(valid_metric)

        elif self.mode =='min':
            best_train_metric_pos = np.argmin(train_metric)
            best_val_metric_pos = np.argmin(valid_metric)

        best_train_metric = train_metric[best_train_metric_pos]
        best_train_loss = train_loss[best_train_metric_pos]

        best_val_metric = valid_metric[best_val_metric_pos]
        best_val_loss = valid_loss[best_val_metric_pos]

        print('=' * 50)
        print('[info msg] training is done')
        print("Time taken: {}".format(self.elapsed_time))
        print("best metric is {} w/ loss {} at epoch : {}".format(best_val_metric, best_val_loss, best_val_metric_pos))    

        print('=' * 50)
        print('[info msg] model weight and log is save to {}'.format(self.save_path))

        with open(os.path.join(self.save_path, 'log.txt'), 'w') as f:
            f.write(f'total ecpochs : {train_loss.shape[0]}\n')
            f.write(f'time taken : {self.elapsed_time}\n')
            f.write(f'best_train_metric {best_train_metric} w/ loss {best_train_loss} at epoch : {best_train_metric_pos}\n')
            f.write(f'best_valid_metric {best_val_metric} w/ loss {best_val_loss} at epoch : {best_val_metric_pos}\n')

        df_learning_curves = pd.DataFrame.from_dict({
                    'loss_train': train_loss,
                    'loss_val': valid_loss,
                    'metric_train': train_metric,
                    'metric_val': valid_metric,
                })

        df_learning_curves.to_csv(os.path.join(self.save_path, 'learning_curves.csv'), sep=',')

        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.title('loss')
        plt.plot(train_loss, label='train loss')
        plt.plot(valid_loss, label='valid loss')
        plt.axvline(x=best_val_metric_pos, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('metric')
        plt.plot(train_metric, label='train metric')
        plt.plot(valid_metric, label='valid metric')
        plt.axvline(x=best_val_metric_pos, color='r', linestyle='--', linewidth=1.5, label='best_val_metric')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'history.png'))
        # plt.show()
        
        plt.figure(figsize=(15,5))
        plt.title('lr_rate curve')
        plt.grid(True)
        plt.plot(self.lr_curve)
        plt.savefig(os.path.join(self.save_path, 'lr_history.png'))
        # plt.show()

    @property
    def save_dir(self):
        return self.save_path