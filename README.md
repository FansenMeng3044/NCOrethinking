# POMO

### Files in "OLD_ipynb_ver" folders 

They are the original codes (2020) used for the paper<br>
> POMO: Policy Optimization with Multiple Optima for Reinforcement Learning<br>
> accepted at NeurIPS 2020<br>
http://arxiv.org/abs/2010.16011
https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html 

They are based on ipynb files and easier to play with interactively.
 <br>
 <br>
 <br>
### Files in "NEW_py_ver" folders 
They are the updated codes, newly structured in 2021. <br>
They are based on py files, so that they can be run on servers more easily.

### Multi-vehicle CVRPTW

The Python implementation now includes AM-TW, POMO-TW, and their xy-only
giant-tour + hard-TW Split variants. See
[`NEW_py_ver/CVRP/CVRPTW_README.md`](NEW_py_ver/CVRP/CVRPTW_README.md) for the
shared route semantics, fixed-data format, training, evaluation, and tests.

### Reproducibility archives

- [`artifacts/cvrp_xml100_20260827`](artifacts/cvrp_xml100_20260827): final
  eight non-TW CVRP checkpoints (n50/n100), full training evidence, the official
  XML100 release, audited evaluation code, 40,000 verified n100 results and
  routes, and plotting-ready figures/scripts.
- [`artifacts/cvrptw_tw_solomon_20260827`](artifacts/cvrptw_tw_solomon_20260827):
  final CVRPTW checkpoints, training records, and Solomon evaluation archive.

