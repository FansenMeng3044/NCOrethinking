import math
import time
from pathlib import Path

import torch

from CVRPTWCore import replay_cvrptw_actions, split_routes_to_actions
from POMO.CVRPTester import CVRPTester
from POMO_SPLIT.GiantTourModel import GiantTourModel
from POMO_SPLIT_TW.GiantTourTWEnv import GiantTourTWEnv


class GiantTourTWTester(CVRPTester):
    """Evaluate the selected POMO-Split-TW solution with strict replay."""

    ENV_CLASS = GiantTourTWEnv
    MODEL_CLASS = GiantTourModel

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
        no_aug_values, aug_values, route_values = [], [], []
        strict_replay_instances = 0
        feasible_instances = 0
        episode = 0
        started = time.time()
        while episode < test_episodes:
            batch_size = min(
                self.tester_params["test_batch_size"], test_episodes - episode
            )
            details = self._test_one_batch_details(batch_size)
            no_aug_values.append(details["distance_1x"])
            aug_values.append(details["distance_aug"])
            route_values.append(details["route_counts"])
            strict_replay_instances += batch_size
            feasible_instances += int(details["feasible"].sum().item())
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

        _, no_aug_mean, no_aug_se = self._stats(no_aug_values)
        augmented, aug_mean, aug_se = self._stats(aug_values)
        routes = torch.cat(route_values).double()
        model_load = self.tester_params["model_load"]
        checkpoint = Path(model_load["path"]) / "checkpoint-{}.pt".format(
            model_load["epoch"]
        )
        result = {
            "problem": "pomo_split_tw",
            "model": str(checkpoint.expanduser().resolve()),
            "dataset": str(Path(
                self.tester_params["test_data_load"]["filename"]
            ).expanduser().resolve()),
            "instances": int(augmented.numel()),
            "augmentation": int(self.tester_params.get("aug_factor", 1)),
            "decode_strategy": "pomo_argmax",
            "pomo_size": int(self.env.pomo_size),
            "mean_distance_1x": no_aug_mean,
            "standard_error_1x": no_aug_se,
            "mean_distance_aug": aug_mean,
            "standard_error_aug": aug_se,
            "mean_vehicle_count": routes.mean().item(),
            "strict_replay_instances": strict_replay_instances,
            "strict_replay_feasible_instances": feasible_instances,
            "all_selected_solutions_feasible": (
                feasible_instances == strict_replay_instances
            ),
            "elapsed_seconds": time.time() - started,
        }
        self.logger.info(
            "Test done: distance=%.6f +- %.6f, vehicles=%.4f, feasible=%d/%d",
            aug_mean, aug_se, routes.mean().item(), feasible_instances,
            strict_replay_instances,
        )
        return result

    def _test_one_batch_details(self, batch_size):
        aug_factor = (
            self.tester_params["aug_factor"]
            if self.tester_params["augmentation_enable"] else 1
        )
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
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

            tours = self.env.selected_node_list.reshape(
                aug_factor, batch_size, self.env.pomo_size, self.env.problem_size
            )
            predecessors = self.env.last_split_result.predecessors.reshape(
                aug_factor, batch_size, self.env.pomo_size,
                self.env.problem_size + 1,
            )
            expected_routes = self.env.last_split_result.route_counts.reshape(
                aug_factor, batch_size, self.env.pomo_size
            )[best_aug, rows, chosen_pomo]
            chosen_tours = tours[best_aug, rows, chosen_pomo]
            chosen_predecessors = predecessors[best_aug, rows, chosen_pomo]
            actions = split_routes_to_actions(chosen_tours, chosen_predecessors)

            def choose(tensor):
                shaped = tensor.reshape(
                    aug_factor, batch_size, *tensor.shape[1:]
                )
                return shaped[best_aug, rows]

            replay = replay_cvrptw_actions(
                choose(self.env.depot_xy),
                choose(self.env.node_xy),
                choose(self.env.node_demand),
                choose(self.env.node_service_time),
                choose(self.env.node_tw_start),
                choose(self.env.node_tw_end),
                actions,
                capacity=self.env.capacity,
                depot_start=self.env.depot_start,
                depot_end=self.env.depot_end,
                speed=self.env.speed,
                loc_scaler=self.env.loc_scaler,
                epsilon=self.env.epsilon,
            )
            distance_matches = torch.isclose(
                replay.distances, distance_aug, rtol=1e-5, atol=1e-5
            )
            route_counts_match = replay.route_counts == expected_routes
            feasible = replay.feasible & distance_matches & route_counts_match

        return {
            "distance_1x": distance_1x.detach().cpu(),
            "distance_aug": distance_aug.detach().cpu(),
            "route_counts": replay.route_counts.detach().cpu(),
            "feasible": feasible.detach().cpu(),
        }
