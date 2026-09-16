# POMO-MTL-Split official MVMoE evaluation

Protocol: official released n=50/n=100 datasets, 1,000 instances per environment, greedy decoding, POMO size n, and 8-fold augmentation. `ok` solutions were independently replayed during evaluation for all active constraints.

A gap is an official full-set gap only when all 1,000 instances in that environment have feasible candidates. Rows with failures report a success-conditional gap and are marked `--` in the full-set gap column.

## Aggregate results

| Model | n | Split | Envs complete | Success | Success rate | Macro cost (ok) | Macro vehicles (ok) | Conditional gap | Full-set gap |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| POMO-MTL-Split n=100 | 100 | IID | 6/6 | 6000/6000 | 100.00% | 18.3919 | 16.418 | 17.894% | 17.894% |
| POMO-MTL-Split n=100 | 100 | OOD | 6/10 | 6035/10000 | 60.35% | 22.2175 | 21.643 | 31.752% | -- |
| POMO-MTL-Split n=100 | 100 | ALL | 12/16 | 12035/16000 | 75.22% | 20.7829 | 19.684 | 26.555% | -- |
| POMO-MTL-Split n=50 | 50 | IID | 6/6 | 6000/6000 | 100.00% | 11.1300 | 9.616 | 12.755% | 12.755% |
| POMO-MTL-Split n=50 | 50 | OOD | 6/10 | 6723/10000 | 67.23% | 12.5018 | 12.244 | 25.960% | -- |
| POMO-MTL-Split n=50 | 50 | ALL | 12/16 | 12723/16000 | 79.52% | 11.9874 | 11.259 | 21.008% | -- |

## Per-environment results

### POMO-MTL-Split n=100 (n=100)

| Environment | Split | Success | Cost (ok) | Vehicles (ok) | Conditional gap | Full-set gap | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| CVRP | IID | 1000/1000 | 16.6911 | 11.098 | 7.622% | 7.622% | 0 |
| OVRP | IID | 1000/1000 | 11.2898 | 12.748 | 14.844% | 14.844% | 0 |
| VRPB | IID | 1000/1000 | 12.6401 | 6.838 | 6.393% | 6.393% | 0 |
| VRPL | IID | 1000/1000 | 16.7515 | 11.159 | 6.188% | 6.188% | 0 |
| VRPTW | IID | 1000/1000 | 34.1056 | 28.110 | 41.128% | 41.128% | 0 |
| OVRPTW | IID | 1000/1000 | 18.8732 | 28.558 | 31.192% | 31.192% | 0 |
| OVRPB | OOD | 1000/1000 | 9.7537 | 7.996 | 16.542% | 16.542% | 0 |
| OVRPL | OOD | 1000/1000 | 11.2510 | 12.790 | 14.923% | 14.923% | 0 |
| VRPBL | OOD | 1000/1000 | 12.5874 | 6.870 | 6.712% | 6.712% | 0 |
| VRPBTW | OOD | 6/1000 | 41.5384 | 32.667 | 53.903% | -- | 994 |
| VRPLTW | OOD | 1000/1000 | 34.1227 | 28.145 | 36.790% | 36.790% | 0 |
| OVRPBL | OOD | 1000/1000 | 9.7417 | 7.929 | 16.632% | 16.632% | 0 |
| OVRPBTW | OOD | 11/1000 | 21.3008 | 30.455 | 47.897% | -- | 989 |
| OVRPLTW | OOD | 1000/1000 | 18.7400 | 28.550 | 31.232% | 31.232% | 0 |
| VRPBLTW | OOD | 8/1000 | 40.3753 | 30.125 | 41.450% | -- | 992 |
| OVRPBLTW | OOD | 10/1000 | 22.7645 | 30.900 | 51.439% | -- | 990 |

### POMO-MTL-Split n=50 (n=50)

| Environment | Split | Success | Cost (ok) | Vehicles (ok) | Conditional gap | Full-set gap | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| CVRP | IID | 1000/1000 | 10.8804 | 7.122 | 5.224% | 5.224% | 0 |
| OVRP | IID | 1000/1000 | 7.2610 | 8.518 | 11.654% | 11.654% | 0 |
| VRPB | IID | 1000/1000 | 8.4041 | 4.433 | 4.394% | 4.394% | 0 |
| VRPL | IID | 1000/1000 | 10.9620 | 7.188 | 4.420% | 4.420% | 0 |
| VRPTW | IID | 1000/1000 | 18.7057 | 14.833 | 29.291% | 29.291% | 0 |
| OVRPTW | IID | 1000/1000 | 10.5668 | 15.600 | 21.544% | 21.544% | 0 |
| OVRPB | OOD | 1000/1000 | 6.5290 | 5.598 | 13.571% | 13.571% | 0 |
| OVRPL | OOD | 1000/1000 | 7.2344 | 8.513 | 11.408% | 11.408% | 0 |
| VRPBL | OOD | 1000/1000 | 8.4268 | 4.482 | 4.887% | 4.887% | 0 |
| VRPBTW | OOD | 163/1000 | 21.7856 | 16.859 | 44.705% | -- | 837 |
| VRPLTW | OOD | 1000/1000 | 18.6134 | 14.840 | 28.130% | 28.130% | 0 |
| OVRPBL | OOD | 1000/1000 | 6.5102 | 5.654 | 13.353% | 13.353% | 0 |
| OVRPBTW | OOD | 194/1000 | 12.1585 | 17.196 | 39.430% | -- | 806 |
| OVRPLTW | OOD | 1000/1000 | 10.5645 | 15.589 | 21.709% | 21.709% | 0 |
| VRPBLTW | OOD | 169/1000 | 21.0859 | 16.627 | 44.431% | -- | 831 |
| OVRPBLTW | OOD | 197/1000 | 12.1097 | 17.086 | 37.979% | -- | 803 |

