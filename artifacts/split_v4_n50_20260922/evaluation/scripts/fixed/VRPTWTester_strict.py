import csv
import math
import time
from pathlib import Path

import torch

from CVRPTWCore import replay_cvrptw_actions
from POMO.CVRPTester import CVRPTester
from POMO_TW.VRPTWEnv import VRPTWEnv
from POMO_TW.VRPTWModel import VRPTWModel


class VRPTWTester(CVRPTester):
    """Evaluate POMO-TW and strictly replay every selected solution."""

    ENV_CLASS = VRPTWEnv
    MODEL_CLASS = VRPTWModel

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _stats(values):
        values = torch.cat(values).double()
        mean = values.mean().item()
        standard_error = (
            (values.std(unbiased=True) / math.sqrt(values.numel())).item()
            if values.numel() > 1 else 0.0
        )
        return values, mean, standard_error

    def run(self):
        self.time_estimator.reset()
        if self.tester_params["test_data_load"]["enable"]:
            self.env.use_saved_problems(
                self.tester_params["test_data_load"]["filename"], self.device
            )

        test_episodes = self.tester_params["test_episodes"]
        warmup_batches = int(self.tester_params.get("timing_warmup_batches", 1))
        if warmup_batches < 0:
            raise ValueError("timing_warmup_batches must be non-negative")
        if warmup_batches:
            warmup_size = min(
                self.tester_params["test_batch_size"], test_episodes
            )
            for _ in range(warmup_batches):
                if self.env.saved_problems is not None:
                    self.env.saved_index = 0
                self._test_one_batch_details(warmup_size)
            if self.env.saved_problems is not None:
                self.env.saved_index = 0
            self._sync_device()
            self.time_estimator.reset()

        no_aug_values, aug_values, route_values, feasible_values = [], [], [], []
        inference_seconds = 0.0
        replay_seconds = 0.0
        episode = 0
        self._sync_device()
        wall_started = time.perf_counter()
        while episode < test_episodes:
            batch_size = min(
                self.tester_params["test_batch_size"], test_episodes - episode
            )
            details = self._test_one_batch_details(batch_size)
            no_aug_values.append(details["distance_1x"])
            aug_values.append(details["distance_aug"])
            route_values.append(details["route_counts"])
            feasible_values.append(details["feasible"])
            inference_seconds += details["inference_seconds"]
            replay_seconds += details["replay_seconds"]
            episode += batch_size
            elapsed, remaining = self.time_estimator.get_est_string(
                episode, test_episodes
            )
            self.logger.info(
                "episode %d/%d, Elapsed[%s], Remain[%s], score:%.4f, "
                "aug_score:%.4f, strict_feasible:%d/%d",
                episode, test_episodes, elapsed, remaining,
                details["distance_1x"].double().mean().item(),
                details["distance_aug"].double().mean().item(),
                int(details["feasible"].sum().item()), batch_size,
            )

        self._sync_device()
        audited_wall_seconds = time.perf_counter() - wall_started
        no_aug, no_aug_mean, no_aug_se = self._stats(no_aug_values)
        augmented, aug_mean, aug_se = self._stats(aug_values)
        routes = torch.cat(route_values).double()
        feasible = torch.cat(feasible_values).bool()
        per_instance_output = self.tester_params.get("per_instance_output")
        if per_instance_output:
            self._write_per_instance(
                per_instance_output, no_aug, augmented, routes, feasible
            )

        model_load = self.tester_params["model_load"]
        checkpoint = Path(model_load["path"]) / "checkpoint-{}.pt".format(
            model_load["epoch"]
        )
        result = {
            "problem": "pomo_tw",
            "model": str(checkpoint.expanduser().resolve()),
            "dataset": str(Path(
                self.tester_params["test_data_load"]["filename"]
            ).expanduser().resolve()),
            "instances": int(augmented.numel()),
            "augmentation": int(self.tester_params.get("aug_factor", 1)),
            "decode_strategy": "pomo_argmax",
            "pomo_size": int(self.env.pomo_size),
            "timing_batch_size": int(self.tester_params["test_batch_size"]),
            "timing_warmup_batches": warmup_batches,
            "mean_distance_1x": no_aug_mean,
            "standard_error_1x": no_aug_se,
            "mean_distance_aug": aug_mean,
            "standard_error_aug": aug_se,
            "mean_vehicle_count": routes.mean().item(),
            "strict_replay_instances": int(feasible.numel()),
            "strict_replay_feasible_instances": int(feasible.sum().item()),
            "all_selected_solutions_feasible": bool(feasible.all().item()),
            "inference_seconds": inference_seconds,
            "strict_replay_seconds": replay_seconds,
            "audited_wall_seconds": audited_wall_seconds,
            "inference_instances_per_second": (
                augmented.numel() / inference_seconds
                if inference_seconds > 0 else None
            ),
            "per_instance_output": (
                str(Path(per_instance_output).expanduser().resolve())
                if per_instance_output else None
            ),
        }
        self.logger.info(
            "Test done: distance=%.6f +- %.6f, vehicles=%.4f, feasible=%d/%d",
            aug_mean, aug_se, routes.mean().item(), int(feasible.sum().item()),
            feasible.numel(),
        )
        return result

    @staticmethod
    def _write_per_instance(path, no_aug, augmented, routes, feasible):
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("instance", "distance_1x", "distance_aug", "vehicle_count", "feasible")
            )
            for index, values in enumerate(zip(no_aug, augmented, routes, feasible)):
                writer.writerow((
                    index, values[0].item(), values[1].item(),
                    int(values[2].item()), int(values[3].item()),
                ))

    def _test_one_batch_details(self, batch_size):
        aug_factor = (
            self.tester_params["aug_factor"]
            if self.tester_params["augmentation_enable"] else 1
        )
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            self._sync_device()
            inference_started = time.perf_counter()
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            costs = (-reward).reshape(aug_factor, batch_size, self.env.pomo_size)
            best_pomo_cost, best_pomo = costs.min(dim=2)
            distance_1x = best_pomo_cost[0]
            distance_aug, best_aug = best_pomo_cost.min(dim=0)
            rows = torch.arange(batch_size, device=self.device)
            chosen_pomo = best_pomo[best_aug, rows]
            actions = self.env.selected_node_list.reshape(
                aug_factor, batch_size, self.env.pomo_size, -1
            )[best_aug, rows, chosen_pomo]
            self._sync_device()
            inference_seconds = time.perf_counter() - inference_started

            def choose(tensor):
                return tensor.reshape(
                    aug_factor, batch_size, *tensor.shape[1:]
                )[best_aug, rows]

            self._sync_device()
            replay_started = time.perf_counter()
            replay = replay_cvrptw_actions(
                choose(self.env.depot_xy), choose(self.env.node_xy),
                choose(self.env.node_demand),
                choose(self.env.node_service_time),
                choose(self.env.node_tw_start), choose(self.env.node_tw_end),
                actions, capacity=self.env.capacity,
                depot_start=self.env.depot_start, depot_end=self.env.depot_end,
                speed=self.env.speed, loc_scaler=self.env.loc_scaler,
                epsilon=self.env.epsilon,
            )
            self._sync_device()
            replay_seconds = time.perf_counter() - replay_started
            distance_matches = torch.isclose(
                replay.distances, distance_aug, rtol=1e-5, atol=1e-5
            )
            feasible = replay.feasible & distance_matches

        return {
            "distance_1x": distance_1x.detach().cpu(),
            "distance_aug": distance_aug.detach().cpu(),
            "route_counts": replay.route_counts.detach().cpu(),
            "feasible": feasible.detach().cpu(),
            "inference_seconds": inference_seconds,
            "replay_seconds": replay_seconds,
        }
