
import time
import traceback

import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from utils.training_metrics import POMOTrainingMetrics


class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        self.metrics_logger = POMOTrainingMetrics(
            self.result_folder, 'pomo_cvrp', self.env_params, self.model_params,
            self.optimizer_params, self.trainer_params, self.model, self.device
        )

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        status = 'completed'
        error = None
        try:
            for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
                epoch_start = time.perf_counter()
                self.metrics_logger.start_epoch(epoch)
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.logger.info('=================================================================')

                # LR Decay
                self.scheduler.step()
                learning_rate_start = self.optimizer.param_groups[0]['lr']

                # Train
                training_start = time.perf_counter()
                train_score, train_loss, global_step_end = self._train_one_epoch(epoch)
                training_seconds = time.perf_counter() - training_start
                self.result_log.append('train_score', epoch, train_score)
                self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
                self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

                all_done = (epoch == self.trainer_params['epochs'])
                model_save_interval = self.trainer_params['logging']['model_save_interval']
                checkpoint_path = ''
                checkpoint_seconds = 0.0

            # Save Model
                if all_done or (epoch % model_save_interval) == 0:
                    self.logger.info("Saving trained_model")
                    checkpoint_start = time.perf_counter()
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                    checkpoint_path = '{}/checkpoint-{}.pt'.format(self.result_folder, epoch)
                    torch.save(checkpoint_dict, checkpoint_path)
                    checkpoint_seconds = time.perf_counter() - checkpoint_start
                    self.metrics_logger.log_checkpoint(epoch, checkpoint_path, checkpoint_seconds)

                epoch_total_seconds = time.perf_counter() - epoch_start
                self.metrics_logger.log_epoch({
                    'epoch': epoch,
                    'global_step_end': global_step_end,
                    'learning_rate_start': learning_rate_start,
                    'learning_rate_end': self.optimizer.param_groups[0]['lr'],
                    'training_seconds': training_seconds,
                    'checkpoint_seconds': checkpoint_seconds,
                    'epoch_total_seconds': epoch_total_seconds,
                    'throughput_instances_per_second': self.trainer_params['train_episodes'] / max(training_seconds, 1e-12),
                    'checkpoint_saved': bool(checkpoint_path),
                    'checkpoint_path': checkpoint_path,
                })

            # All-done announcement
                if all_done:
                    self.logger.info(" *** Training Done *** ")
                    self.logger.info("Now, printing log array...")
                    util_print_log_array(self.logger, self.result_log)
        except BaseException as exc:
            status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            error = traceback.format_exc()
            raise
        finally:
            self.metrics_logger.close(status=status, error=error)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        batches_per_epoch = (train_num_episode + self.trainer_params['train_batch_size'] - 1) // self.trainer_params['train_batch_size']
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            batch_metrics = self._train_one_batch(batch_size)
            batch_metrics.update({
                'epoch': epoch,
                'batch_id': loop_cnt,
                'global_step': (epoch - 1) * batches_per_epoch + loop_cnt,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            })
            self.metrics_logger.log_batch(batch_metrics)
            avg_score, avg_loss = batch_metrics['score_mean'], batch_metrics['loss_mean']
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            loop_cnt += 1
            if epoch == self.start_epoch:
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg, epoch * batches_per_epoch

    def _train_one_batch(self, batch_size):

        batch_start = time.perf_counter()

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        grad_parts = [p.grad.detach().float().norm(2) for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.stack(grad_parts).norm(2) if grad_parts else loss_mean.new_tensor(0.0)
        self.optimizer.step()

        score_values = -max_pomo_reward.detach().float()
        loss_values = loss.detach().float().mean(dim=1)
        packed = torch.stack([
            score_values.mean(), score_values.std(unbiased=False), (-reward.detach().float()).mean(),
            loss_values.mean(), loss_values.std(unbiased=False), advantage.detach().float().mean(),
            advantage.detach().float().std(unbiased=False), log_prob.detach().float().mean(),
            grad_norm, score_values.sum(), score_values.square().sum(), loss_values.sum(),
            loss_values.square().sum(),
        ]).cpu().tolist()
        (score_value, score_std, solution_cost_mean, loss_value, loss_std, advantage_mean,
         advantage_std, log_prob_mean, grad_norm_value, score_sum, score_sumsq,
         loss_sum, loss_sumsq) = packed
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0.0
        step_seconds = time.perf_counter() - batch_start
        return {
            'batch_size': batch_size, 'score_mean': score_value, 'score_std': score_std,
            'solution_cost_mean': solution_cost_mean, 'loss_mean': loss_value,
            'loss_std': loss_std, 'advantage_mean': advantage_mean,
            'advantage_std': advantage_std, 'log_prob_mean': log_prob_mean,
            'nll_mean': -log_prob_mean, 'grad_norm': grad_norm_value,
            'step_seconds': step_seconds,
            'throughput_instances_per_second': batch_size / max(step_seconds, 1e-12),
            'gpu_memory_allocated_mb': allocated, 'gpu_memory_reserved_mb': reserved,
            'gpu_peak_allocated_mb': peak_allocated, 'gpu_peak_reserved_mb': peak_reserved,
            '_score_sum': score_sum, '_score_sumsq': score_sumsq,
            '_loss_sum': loss_sum, '_loss_sumsq': loss_sumsq,
        }
