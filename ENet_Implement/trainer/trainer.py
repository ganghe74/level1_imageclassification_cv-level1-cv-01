import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        
        wandb.init(project="Mask-Classification", entity="yon-ninii") # wandb initialization
        wandb.config = { # wandb configuration
        "learning_rate": config['optimizer']['args']['lr'],
        "epochs": config['trainer']['epochs'],
        "batch_size": config['data_loader']['args']['batch_size']
        }
        
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('Train_loss', *['Train_' + m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('Val_loss', *['Val_' + m.__name__ for m in self.metric_ftns], writer=self.writer)
        #wandb.watch(self.model, self.criterion, log='all', log_freq=1)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('Train_loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update('Train_' + met.__name__, met(output, target, self.device))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        '''
        train_acc = log['accuracy']
        train_loss = log['loss']
        wandb.log({'train_acc':train_acc, 'train_loss':train_loss})
        '''
        
        log = self.train_metrics.result() # Dict type
        #wandb.log(log)
        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        wandb.log(log)
        
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('Val_loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update('Val_' + met.__name__, met(output, target, self.device))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
            
        val_log = self.valid_metrics.result()
        #wandb.log(val_log)
        '''
        val_acc = val_log['accuracy']
        val_loss = val_log['loss']
        wandb.log({'val_acc':val_acc, 'val_loss':val_loss})
        '''
        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
