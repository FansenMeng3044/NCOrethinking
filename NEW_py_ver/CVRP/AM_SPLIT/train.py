import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, return_details=False):
    # Validate
    print('Validating...')
    start_time = time.perf_counter()
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    count = len(cost)
    std_cost = torch.std(cost, unbiased=count > 1)
    standard_error = std_cost / math.sqrt(count)
    duration = time.perf_counter() - start_time
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, standard_error))

    if return_details:
        return {
            'validation_count': count,
            'validation_cost_mean': avg_cost.item(),
            'validation_cost_std': std_cost.item(),
            'validation_standard_error': standard_error.item(),
            'validation_seconds': duration,
        }
    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts,
                metrics_logger=None):
    completed_epoch = epoch + 1
    learning_rate_start = optimizer.param_groups[0]['lr']
    print("Start train epoch {}, lr={} for run {}".format(completed_epoch, learning_rate_start, opts.run_name))
    batches_per_epoch = (opts.epoch_size + opts.batch_size - 1) // opts.batch_size
    step = epoch * batches_per_epoch
    epoch_start_time = time.perf_counter()
    if metrics_logger is not None:
        metrics_logger.start_epoch(completed_epoch)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', learning_rate_start, step)

    # Generate new training data for each epoch
    prepare_start_time = time.perf_counter()
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    data_preparation_seconds = time.perf_counter() - prepare_start_time

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    if opts.use_cuda:
        torch.cuda.reset_peak_memory_stats(opts.device)

    training_start_time = time.perf_counter()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        batch_metrics = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        if metrics_logger is not None:
            metrics_logger.log_batch(batch_metrics)

        step += 1
    training_seconds = time.perf_counter() - training_start_time

    print("Finished training epoch {}, took {} s".format(
        completed_epoch, time.strftime('%H:%M:%S', time.gmtime(training_seconds))))

    checkpoint_path = ''
    checkpoint_seconds = 0.0
    if (
        opts.checkpoint_epochs != 0
        and completed_epoch % opts.checkpoint_epochs == 0
    ) or completed_epoch == opts.n_epochs:
        print('Saving model and state...')
        checkpoint_start_time = time.perf_counter()
        checkpoint_path = os.path.join(opts.save_dir, 'epoch-{}.pt'.format(completed_epoch))
        torch.save(
            {
                'epoch': completed_epoch,
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            checkpoint_path
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_start_time
        if metrics_logger is not None:
            metrics_logger.log_checkpoint(completed_epoch, checkpoint_path, checkpoint_seconds)

    validation_metrics = validate(model, val_dataset, opts, return_details=True)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', validation_metrics['validation_cost_mean'], step)

    baseline_callback_start_time = time.perf_counter()
    baseline.epoch_callback(model, epoch)
    baseline_callback_seconds = time.perf_counter() - baseline_callback_start_time
    baseline_metrics = baseline.get_epoch_metrics()

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    epoch_total_seconds = time.perf_counter() - epoch_start_time

    if metrics_logger is not None:
        metrics_logger.log_epoch({
            'epoch': completed_epoch,
            'global_step_end': step,
            'learning_rate_start': learning_rate_start,
            'learning_rate_end': optimizer.param_groups[0]['lr'],
            'data_preparation_seconds': data_preparation_seconds,
            'training_seconds': training_seconds,
            'checkpoint_seconds': checkpoint_seconds,
            'validation_seconds': validation_metrics['validation_seconds'],
            'baseline_callback_seconds': baseline_callback_seconds,
            'epoch_total_seconds': epoch_total_seconds,
            'throughput_instances_per_second': len(training_dataset) / max(training_seconds, 1e-12),
            'validation_count': validation_metrics['validation_count'],
            'validation_cost_mean': validation_metrics['validation_cost_mean'],
            'validation_cost_std': validation_metrics['validation_cost_std'],
            'validation_standard_error': validation_metrics['validation_standard_error'],
            'checkpoint_saved': bool(checkpoint_path),
            'checkpoint_path': checkpoint_path,
        }, baseline_metrics=baseline_metrics)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    batch_start_time = time.perf_counter()
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

    cost_values = cost.detach().float()
    advantage_values = (cost - bl_val).detach().float()
    batch_size = cost_values.numel()
    grad_values, clipped_grad_values = grad_norms

    def as_scalar_tensor(value):
        if torch.is_tensor(value):
            return value.detach().float().reshape(())
        return cost_values.new_tensor(float(value))

    # Transfer all scalar metrics together so detailed CSV logging adds only one GPU sync per batch.
    packed = torch.stack([
        cost_values.mean(),
        cost_values.std(unbiased=False),
        as_scalar_tensor(reinforce_loss),
        as_scalar_tensor(bl_loss),
        as_scalar_tensor(loss),
        log_likelihood.detach().float().mean(),
        as_scalar_tensor(bl_val.detach().float().mean() if torch.is_tensor(bl_val) else bl_val),
        advantage_values.mean(),
        advantage_values.std(unbiased=False),
        as_scalar_tensor(grad_values[0]),
        as_scalar_tensor(clipped_grad_values[0]),
        as_scalar_tensor(grad_values[1]) if len(grad_values) > 1 else cost_values.new_tensor(float('nan')),
        as_scalar_tensor(clipped_grad_values[1]) if len(clipped_grad_values) > 1 else cost_values.new_tensor(float('nan')),
        cost_values.sum(),
        cost_values.square().sum(),
        advantage_values.sum(),
        advantage_values.square().sum(),
    ]).cpu().tolist()
    (
        cost_mean, cost_std, reinforce_loss_value, baseline_loss_value, total_loss_value,
        log_likelihood_mean, baseline_value_mean, advantage_mean, advantage_std, grad_norm,
        grad_norm_clipped, critic_grad_norm, critic_grad_norm_clipped, cost_sum, cost_sumsq,
        advantage_sum, advantage_sumsq,
    ) = packed

    if opts.use_cuda:
        allocated_mb = torch.cuda.memory_allocated(opts.device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(opts.device) / (1024 ** 2)
        peak_allocated_mb = torch.cuda.max_memory_allocated(opts.device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(opts.device) / (1024 ** 2)
    else:
        allocated_mb = reserved_mb = peak_allocated_mb = peak_reserved_mb = 0.0
    step_seconds = time.perf_counter() - batch_start_time

    return {
        'epoch': epoch + 1,
        'batch_id': batch_id,
        'global_step': step,
        'batch_size': batch_size,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'cost_mean': cost_mean,
        'cost_std': cost_std,
        'reinforce_loss': reinforce_loss_value,
        'baseline_loss': baseline_loss_value,
        'total_loss': total_loss_value,
        'log_likelihood_mean': log_likelihood_mean,
        'nll_mean': -log_likelihood_mean,
        'baseline_value_mean': baseline_value_mean,
        'advantage_mean': advantage_mean,
        'advantage_std': advantage_std,
        'grad_norm': grad_norm,
        'grad_norm_clipped': grad_norm_clipped,
        'critic_grad_norm': critic_grad_norm if len(grad_values) > 1 else '',
        'critic_grad_norm_clipped': critic_grad_norm_clipped if len(clipped_grad_values) > 1 else '',
        'step_seconds': step_seconds,
        'throughput_instances_per_second': batch_size / max(step_seconds, 1e-12),
        'gpu_memory_allocated_mb': allocated_mb,
        'gpu_memory_reserved_mb': reserved_mb,
        'gpu_peak_allocated_mb': peak_allocated_mb,
        'gpu_peak_reserved_mb': peak_reserved_mb,
        '_cost_sum': cost_sum,
        '_cost_sumsq': cost_sumsq,
        '_advantage_sum': advantage_sum,
        '_advantage_sumsq': advantage_sumsq,
    }
