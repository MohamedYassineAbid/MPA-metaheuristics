# MPA Metaheuristics

A comprehensive implementation of the **Marine Predators Algorithm (MPA)**, a nature-inspired metaheuristic optimization algorithm based on the foraging behavior of marine predators.

## 📋 Overview

The Marine Predators Algorithm (MPA) is a population-based optimization algorithm that mimics the optimal foraging strategy and encounter rate between predators and prey in marine ecosystems. This repository contains the implementation and benchmarking results of MPA on the CEC2017 benchmark functions.

## 🐋 About Marine Predators Algorithm

MPA is inspired by the Lévy and Brownian movements of marine predators. The algorithm simulates three main strategies:

1. **High-velocity ratio** - Predator moves faster than prey (exploration phase)
2. **Similar velocity** - Predator and prey move at similar speeds (exploitation phase)
3. **Low-velocity ratio** - Prey moves faster than predator (transitional phase)

### Pseudo Code

```
Algorithm: Marine Predators Algorithm (MPA)
Input: Population size (n), Maximum iterations (Max_Iter), Problem dimension (dim)
Output: Best solution and its fitness value

1. Initialize:
   - Create random population of n search agents (prey)
   - Create Elite matrix (top predators)
   - Set FADs = 0.2, P = 0.5

2. While (iteration < Max_Iter):
   
   a. Construct Elite matrix from top predators
   
   b. For each search agent (prey):
      
      // Phase 1: High velocity ratio (Exploration - Iter < Max_Iter/3)
      If iteration < Max_Iter/3:
         Update prey position using Lévy flight:
         Prey_i = Elite_i + P × R × (Elite_i - Prey_i)
         Update predator using Brownian motion:
         Elite_i = Elite_i + P × CF × (Elite_i - Prey_i)
      
      // Phase 2: Equal velocity ratio (Transition - Max_Iter/3 < Iter < 2×Max_Iter/3)
      Else If iteration < 2×Max_Iter/3:
         Half population - Lévy flight:
         Prey_i = Prey_i + P × R × (Elite_i - Prey_i)
         Other half - Brownian motion:
         Prey_i = Elite_i + P × CF × (R × Elite_i - Prey_i)
      
      // Phase 3: Low velocity ratio (Exploitation - Iter > 2×Max_Iter/3)
      Else:
         Update using Lévy flight:
         Prey_i = Prey_i + P × CF × (Elite_i - Prey_i)
   
   c. Apply Fish Aggregating Devices (FADs) effect:
      For random solutions:
         If U < FADs:
            Prey_i = Prey_i + CF × [Xmin + R × (Xmax - Xmin)] × U
         Else:
            Prey_i = Prey_i + [FADs × (1-r) + r] × (Prey_r1 - Prey_r2)
   
   d. Apply Marine Memory (eddy formation and boundary effect):
      If Prey_i goes beyond search space:
         Prey_i = boundary value
   
   e. Evaluate fitness of all prey
   
   f. Update Elite matrix with best solutions
   
   g. iteration = iteration + 1

3. Return best solution found (top predator) and its fitness
```

**Key Parameters:**
- `FADs` (Fish Aggregating Devices): Controls diversification (typically 0.2)
- `P`: Constant equal to 0.5
- `CF`: Adaptive parameter = (1 - iter/Max_Iter)^(2×iter/Max_Iter)
- `R`: Random vector in [0,1]
- `U`: Random number in [0,1]

## 📊 Benchmark Results

The algorithm has been tested on the CEC2017 benchmark suite (30 functions). Results are stored in `MPA_CEC2017_Results.csv` showing:
- **Mean fitness**: Average performance over multiple runs
- **Standard deviation**: Consistency of results
- **Best fitness**: Best solution found

### Sample Results (CEC2017)


## 🚀 Upcoming Features

### Streamlit Web Application
A user-friendly web interface is currently under development by **Yassine Abid** and **Ahmed Chakcha**. The app will feature:

- 🎯 Interactive parameter tuning
- 📈 Real-time optimization visualization
- 📊 Performance comparison with other algorithms
- 🔧 Custom objective function input
- 📥 Export results and plots
- 🎨 Interactive convergence curves

**Status**: Coming Soon

## 👥 Contributors

- **Mohamed Yassine Abid**
- **Ahmed Chakcha** 

## 📚 References

Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020). *Marine Predators Algorithm: A nature-inspired metaheuristic*. Expert Systems with Applications, 152, 113377.

## 📝 License

This project is open source and available for research and educational purposes.

---

**Note**: This is an active research project. More features and improvements will be added regularly.
