# XML100 overall results

Success is reported before conditional quality metrics.

| Model | Architecture | Aug success | Aug mean Gap % | Aug P95 Gap % | No-aug success | No-aug mean Gap % | Mean vehicles | Inference s/instance | Postprocess s/instance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| am_direct_n50 | am | 10000/10000 | 9.2617 | 18.5653 | 10000/10000 | 16.9402 | 12.245 | 0.001061 | 0.000659 |
| am_split_v4_n50 | am_split | 10000/10000 | 12.2882 | 19.2728 | 10000/10000 | 16.4084 | 12.973 | 0.000709 | 0.002675 |
| pomo_direct_n50 | pomo | 10000/10000 | 5.8917 | 11.9973 | 10000/10000 | 9.2247 | 12.240 | 0.005403 | 0.002069 |
| pomo_split_v4_n50 | pomo_split | 10000/10000 | 11.7461 | 18.3150 | 10000/10000 | 14.7888 | 12.754 | 0.003251 | 0.033876 |
