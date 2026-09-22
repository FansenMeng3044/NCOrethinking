# AM-Split v4, training size 100: formal evaluation

All reported CVRP runs use the final epoch-100 checkpoint with SHA-256
`58cdc39a2132e475848f48a0908ba3afe0e07720fb59fb6b50809a1e2b53170b`.
The matched AM Direct checkpoint has SHA-256
`450c46de4a257605d66b5705cb38f0fd265ec3587a2c7bd866d0605ee463dfce`.

| Evaluation | Instances | AM Direct cost | AM-Split cost | Split change | Direct wins | Split wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed CVRP100, greedy x8 | 10,000 | 16.308432 | 17.221738 | +5.600% | 9,987 | 13 | 0 |
| CVRPLIB XML100, greedy x8 | 10,000 | 18,391.5513 | 18,890.5438 | +2.713% | 8,435 | 1,557 | 8 |
| CVRP200, greedy, no augmentation | 1,000 | 29.891768 | 31.802383 | +6.392% | 999 | 1 | 0 |
| CVRP500, greedy, no augmentation | 1,000 | 69.957706 | 81.887730 | +17.053% | 1,000 | 0 | 0 |
| CVRP1000, greedy, no augmentation | 1,000 | 150.810875 | 164.347209 | +8.976% | 875 | 125 | 0 |

## Additional statistics

| Evaluation | Direct vehicles | Split vehicles | Direct time/instance | Split time/instance |
|---|---:|---:|---:|---:|
| XML100 | 12.7066 | 12.9106 | 0.002900 s | 0.001967 s |
| CVRP200 | 20.927 | 22.310 | 0.007158 s | 0.007146 s |
| CVRP500 | 52.551 | 55.540 | 0.063163 s | 0.062927 s |
| CVRP1000 | 130.240 | 112.785 | 0.067375 s | 0.497437 s |

The mean XML100 optimality gap is 8.7273% for AM Direct and 12.0216% for
AM-Split. All 20,000 XML100 model-instance records have status `ok`, and an
independent verifier checked customer coverage, capacity feasibility, and the
official integer per-edge `EUC_2D` objective for every saved route. All size
generalization runs contain 1,000 independently replayed feasible solutions.

Solomon is a CVRPTW benchmark and therefore is not evaluated with this CVRP
checkpoint. The corresponding Solomon-100 comparison must use the final
AM-Split CVRPTW100 checkpoint so that demand, service time, hard time windows,
and the depot horizon retain their intended semantics.
