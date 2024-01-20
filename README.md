# Sparse-Ising-Solver

This repository is a GPU-accelerated solver for optimizing Ising models using the Simulated Bifurcation (SB) algorithm. 

This solver was optimized for real-life instances, with a geometric mean (average) speedup of 14.5x and a maximum speedup of 58.1x on sparse graphs over dense models (Ageron, R., Bouquet, T., & Pugliese, L. (2023). Simulated Bifurcation (SB) algorithm for Python (1.2.0)). 

Additionally has a 3.3x geometric mean (average) speedup and a maximum 2,318.6x speedup over the state-of-the-art implementation by Goto et al. (Goto, H., Endo, K., Suzuki, M., et al. (2021). High-performance combinatorial optimization based on classical mechanics.)

To compile this program, run:

```
    mkdir build
    make
```

To run the program, there are two different modes. To run it with custom settings, run:

```
    ./build/program <default?> <.mtx/.tsp> <output file> <maxcut/tsp> <update_window?> <fused?> <random?> <time_step> <max_steps> <num_agents>
```

Where:

  - default: 0 or 1. Determines if it uses default values or custom values.

  - .mtx/.tsp: the input file for the MaxCut or TSP instance, where .mtx is a Market Matrix file for optimizing a MaxCut instance, and .tsp is a Traveling Salesman Problem input file.

  - output file: the output file for the result and other data

  - maxcut/tsp: either "maxcut" or "tsp" depending on the problem

  - update_window: 0 or 1, default is 0. 
    
    -  For MaxCut problems: checks every 50 steps whether an agent has bifurcated or not. Once an agent has bifurcated, its spins will lock onto its bifurcated state and stop updating.
    
    -  For TSP problems: checks every 50 steps whether an agent has a valid solution. Once it achieves a valid solution, its spins will lock onto the valid state and stop updating.

  - fused: 0 or 1, default is 1. If true, the solver will use the fused kernels during computation. The fused kernels are faster, more robust to larger step sizes, and have higher solution qualities, but sometimes have unpredictable behavior. In most cases is preferred.

  - random: 0 or 1, default is 0. If true, the solver will compare the optimized solution to a random solution to determine how well it performed.

  - time_step: a float, default is 0.1. Determines how much time passes between every step. Lower time steps are slower with higher solution qualities, and higher time steps are faster but with erratic behavior. Fused kernel can handle larger time step sizes.

  - max_steps: an integer, default is 10'000. Determines how many iterations the solver takes when solving a problem instance. For larger problem sizes, max_steps may need to be increased.

  - agents: an integer, default is 1024. Determines the number of "agents", or parallel instances that are solved simultaneously. More agents increase the probability of reaching a better solution, and when combined with the fused kernels, increase the solver's resilience to higher time steps. Too many agents will slow down the solver.



To run it with default settings, run:

```
    ./program <default?> <.mtx/.tsp> <output file>
```


This is currently being adapted into a Python package.
