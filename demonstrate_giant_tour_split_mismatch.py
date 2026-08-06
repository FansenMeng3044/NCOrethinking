"""Reproduce two small counterexamples for giant-tour CVRP reasoning.

This script is independent of POMO and SplitDecoder. It exhaustively enumerates
all customer permutations and all capacity-feasible contiguous partitions.
"""

import itertools
import json
import math


CAPACITY = 1.0
EPSILON = 1e-10


def euclidean(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def raw_depot_tour_cost(depot, nodes, permutation):
    cost = euclidean(depot, nodes[permutation[0]])
    cost += sum(
        euclidean(nodes[permutation[index]], nodes[permutation[index + 1]])
        for index in range(len(permutation) - 1)
    )
    return cost + euclidean(nodes[permutation[-1]], depot)


def optimal_contiguous_split(depot, nodes, demands, permutation):
    """Classical O(n^2) Bellman Split for one fixed permutation."""

    n = len(permutation)
    labels = [math.inf] * (n + 1)
    predecessors = [-1] * (n + 1)
    labels[0] = 0.0

    for begin in range(n):
        load = 0.0
        route_cost = 0.0
        for end in range(begin, n):
            customer = permutation[end]
            load += demands[customer]
            if load > CAPACITY + EPSILON:
                break
            if end == begin:
                route_cost = 2 * euclidean(depot, nodes[customer])
            else:
                previous = permutation[end - 1]
                route_cost += euclidean(nodes[previous], nodes[customer])
                route_cost += euclidean(nodes[customer], depot)
                route_cost -= euclidean(nodes[previous], depot)
            candidate = labels[begin] + route_cost
            if candidate < labels[end + 1] - EPSILON:
                labels[end + 1] = candidate
                predecessors[end + 1] = begin

    routes = []
    end = n
    while end > 0:
        begin = predecessors[end]
        if begin < 0:
            raise AssertionError("No feasible Split predecessor")
        routes.append(permutation[begin:end])
        end = begin
    routes.reverse()
    return labels[n], tuple(routes)


def one_based_tour(permutation):
    return [customer + 1 for customer in permutation]


def one_based_routes(routes):
    return [one_based_tour(route) for route in routes]


def enumerate_orders(depot, nodes, demands):
    rows = []
    for permutation in itertools.permutations(range(len(nodes))):
        split_cost, routes = optimal_contiguous_split(
            depot, nodes, demands, permutation
        )
        rows.append(
            {
                "permutation": permutation,
                "raw_cost": raw_depot_tour_cost(depot, nodes, permutation),
                "split_cost": split_cost,
                "routes": routes,
            }
        )
    return rows


def raw_vs_split_counterexample():
    depot = (0.808105754993457, 0.5805301112913348)
    nodes = [
        (0.10243833102315136, 0.5245075202990308),
        (0.5897281078399887, 0.8987292632848224),
        (0.3516206819982449, 0.8940270713719484),
        (0.29508200322079103, 0.29729979452107935),
        (0.265872268410114, 0.8271973742991678),
    ]
    demands = [0.65, 0.35, 0.65, 0.55, 0.45]
    rows = enumerate_orders(depot, nodes, demands)

    minimum_raw = min(row["raw_cost"] for row in rows)
    raw_optima = [
        row for row in rows if row["raw_cost"] <= minimum_raw + EPSILON
    ]
    # Be maximally fair to the raw-tour objective: among every raw-tour optimum,
    # select the one whose post-Split cost happens to be best.
    raw_choice = min(raw_optima, key=lambda row: row["split_cost"])
    split_choice = min(rows, key=lambda row: row["split_cost"])
    penalty = raw_choice["split_cost"] / split_choice["split_cost"] - 1

    return {
        "depot": depot,
        "nodes": nodes,
        "demands": demands,
        "raw_optimal_order_count": len(raw_optima),
        "best_raw_optimal_tour": one_based_tour(raw_choice["permutation"]),
        "raw_tour_cost": raw_choice["raw_cost"],
        "post_split_cost_of_raw_optimum": raw_choice["split_cost"],
        "post_split_routes_of_raw_optimum": one_based_routes(raw_choice["routes"]),
        "globally_split_optimal_tour": one_based_tour(split_choice["permutation"]),
        "globally_best_post_split_cost": split_choice["split_cost"],
        "globally_best_routes": one_based_routes(split_choice["routes"]),
        "raw_objective_post_split_penalty_percent": 100 * penalty,
    }


def demand_blind_counterexample():
    depot = (0.953099170959056, 0.7769021715804781)
    nodes = [
        (0.9438015689865051, 0.061381066087920155),
        (0.8666662630749697, 0.43540254273878964),
        (0.8433535927723921, 0.6807307663339633),
        (0.978156552343762, 0.14871036363517343),
        (0.6919992002946668, 0.5693155605688967),
    ]
    demand_a = [0.35, 0.45, 0.75, 0.65, 0.45]
    demand_b = [0.55, 0.35, 0.75, 0.55, 0.35]
    rows_a = enumerate_orders(depot, nodes, demand_a)
    rows_b = enumerate_orders(depot, nodes, demand_b)

    best_a = min(row["split_cost"] for row in rows_a)
    best_b = min(row["split_cost"] for row in rows_b)
    optima_a = [row for row in rows_a if row["split_cost"] <= best_a + EPSILON]
    optima_b = [row for row in rows_b if row["split_cost"] <= best_b + EPSILON]
    orders_a = {row["permutation"] for row in optima_a}
    orders_b = {row["permutation"] for row in optima_b}

    lookup_a = {row["permutation"]: row for row in rows_a}
    lookup_b = {row["permutation"]: row for row in rows_b}
    best_a_order_under_b = min(lookup_b[order]["split_cost"] for order in orders_a)
    best_b_order_under_a = min(lookup_a[order]["split_cost"] for order in orders_b)
    representative_a = min(optima_a, key=lambda row: lookup_b[row["permutation"]]["split_cost"])
    representative_b = min(optima_b, key=lambda row: lookup_a[row["permutation"]]["split_cost"])

    return {
        "depot": depot,
        "nodes": nodes,
        "demand_a": demand_a,
        "demand_b": demand_b,
        "optimal_order_count_a": len(orders_a),
        "optimal_order_count_b": len(orders_b),
        "common_optimal_order_count": len(orders_a.intersection(orders_b)),
        "best_cost_a": best_a,
        "representative_optimal_tour_a": one_based_tour(
            representative_a["permutation"]
        ),
        "representative_routes_a": one_based_routes(representative_a["routes"]),
        "best_a_order_cost_under_b": best_a_order_under_b,
        "a_to_b_penalty_percent": 100 * (best_a_order_under_b / best_b - 1),
        "best_cost_b": best_b,
        "representative_optimal_tour_b": one_based_tour(
            representative_b["permutation"]
        ),
        "representative_routes_b": one_based_routes(representative_b["routes"]),
        "best_b_order_cost_under_a": best_b_order_under_a,
        "b_to_a_penalty_percent": 100 * (best_b_order_under_a / best_a - 1),
    }


def main():
    report = {
        "raw_tour_vs_post_split": raw_vs_split_counterexample(),
        "same_coordinates_different_demands": demand_blind_counterexample(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
