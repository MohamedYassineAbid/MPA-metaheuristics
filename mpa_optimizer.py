"""
Marine Predators Algorithm Implementation
Based on: A. Faramarzi et al., Expert Systems with Applications, 2020
"""
import numpy as np
import math
# Small helper to adapt CEC functions to batch input
def evaluate(func, X):
    # CEC2017 functions expect 2D input: (N, D) where N is number of samples
    # Ensure X is always 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Call function and ensure output is 1D array
    vals = np.asarray(func(X))
    return vals.flatten()
# Levy flight helper (Mantegna's algorithm)
def levy_flight(D, beta=1.5):
    """Generate a single Levy flight step vector"""
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=D)
    v = np.random.normal(0, 1, size=D)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step

def levy_flight_vector(N, D, beta=1.5):
    """Generate Levy flight matrix for N agents with D dimensions"""
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=(N, D))
    v = np.random.normal(0, 1, size=(N, D))
    step = u / (np.abs(v) ** (1.0 / beta))
    return step
# Marine Predators Algorithm (MPA) - Official implementation from paper
# Reference: A. Faramarzi et al., Expert Systems with Applications, 2020
# DOI: doi.org/10.1016/j.eswa.2020.113377
def mpa(func, rng, D=30, N=30, LB=-100, UB=100, Tmax=1000, FADs=0.2, P=0.5):
    """
    Marine Predators Algorithm

    Parameters:
    - func: objective function to minimize
    - rng: numpy random generator
    - D: dimension
    - N: number of search agents (prey)
    - LB, UB: lower and upper bounds
    - Tmax: maximum iterations
    - FADs: Fish Aggregating Devices effect probability (default 0.2)
    - P: constant (default 0.5)
    """
    # Initialize prey population
    Prey = rng.uniform(LB, UB, size=(N, D))

    # Bounds arrays
    Xmin = np.tile(LB * np.ones(D), (N, 1))
    Xmax = np.tile(UB * np.ones(D), (N, 1))

    # Initialize top predator
    Top_predator_pos = np.zeros(D)
    Top_predator_fit = np.inf

    # Initialize fitness and history
    fitness = np.full(N, np.inf)
    Convergence_curve = np.zeros(Tmax)
    stepsize = np.zeros((N, D))

    # Marine Memory
    Prey_old = Prey.copy()
    fit_old = fitness.copy()

    for Iter in range(Tmax):
        # Detecting top predator
        for i in range(N):
            # Boundary checking
            Flag4ub = Prey[i, :] > UB
            Flag4lb = Prey[i, :] < LB
            Prey[i, :] = (Prey[i, :] * (~(Flag4ub | Flag4lb)) +
                         UB * Flag4ub + LB * Flag4lb)

            # Fitness evaluation
            fitness[i] = evaluate(func, Prey[i:i+1, :])[0]

            # Update top predator
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :].copy()

        # Marine Memory saving
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = Prey.copy()

        Inx = fit_old < fitness
        Indx = np.tile(Inx[:, np.newaxis], (1, D))
        Prey = Indx * Prey_old + ~Indx * Prey
        fitness = Inx * fit_old + ~Inx * fitness

        fit_old = fitness.copy()
        Prey_old = Prey.copy()

        # Elite matrix (Eq. 10)
        Elite = np.tile(Top_predator_pos, (N, 1))

        # CF calculation
        CF = (1 - Iter / Tmax) ** (2 * Iter / Tmax)

        # Levy and Brownian random vectors
        RL = 0.05 * levy_flight_vector(N, D, 1.5)
        RB = rng.standard_normal((N, D))

        # Update prey positions based on phase
        for i in range(N):
            for j in range(D):
                R = rng.random()

                # Phase 1: High velocity ratio (Eq. 12)
                if Iter < Tmax / 3:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]

                # Phase 2: Unit velocity ratio (Eqs. 13 & 14)
                elif Iter < 2 * Tmax / 3:
                    if i > N / 2:
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]

                # Phase 3: Low velocity ratio (Eq. 15)
                else:
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]

        # Detecting top predator (after movement)
        for i in range(N):
            # Boundary checking
            Flag4ub = Prey[i, :] > UB
            Flag4lb = Prey[i, :] < LB
            Prey[i, :] = (Prey[i, :] * (~(Flag4ub | Flag4lb)) +
                         UB * Flag4ub + LB * Flag4lb)

            # Fitness evaluation
            fitness[i] = evaluate(func, Prey[i:i+1, :])[0]

            # Update top predator
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :].copy()

        # Marine Memory saving (second time)
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = Prey.copy()

        Inx = fit_old < fitness
        Indx = np.tile(Inx[:, np.newaxis], (1, D))
        Prey = Indx * Prey_old + ~Indx * Prey
        fitness = Inx * fit_old + ~Inx * fitness

        fit_old = fitness.copy()
        Prey_old = Prey.copy()

        # Eddy formation and FADs effect (Eq. 16)
        if rng.random() < FADs:
            U = rng.random((N, D)) < FADs
            Prey = Prey + CF * ((Xmin + rng.random((N, D)) * (Xmax - Xmin)) * U)
        else:
            r = rng.random()
            Rs = N
            # Random permutations for FADs effect
            idx1 = rng.permutation(Rs)
            idx2 = rng.permutation(Rs)
            stepsize = (FADs * (1 - r) + r) * (Prey[idx1, :] - Prey[idx2, :])
            Prey = Prey + stepsize

        # Save convergence
        Convergence_curve[Iter] = Top_predator_fit

    return Convergence_curve , Top_predator_fit, Top_predator_pos


def mpa_with_callback(func, D=30, N=30, LB=-100, UB=100, Tmax=1000, FADs=0.2, P=0.5, callback=None):
    """
    Wrapper for MPA with callback support for real-time visualization
    
    Parameters:
    - func: objective function to minimize
    - D: dimension
    - N: number of search agents (prey)
    - LB, UB: lower and upper bounds
    - Tmax: maximum iterations
    - FADs: Fish Aggregating Devices effect probability (default 0.2)
    - P: constant (default 0.5)
    - callback: optional function(iter, best_fitness) called each iteration
    
    Returns:
    - Top_predator_pos: best solution found
    - Top_predator_fit: best fitness value
    - Convergence_curve: fitness history over iterations
    """
    rng = np.random.default_rng()
    
    # Initialize prey population
    Prey = rng.uniform(LB, UB, size=(N, D))

    # Bounds arrays
    Xmin = np.tile(LB * np.ones(D), (N, 1))
    Xmax = np.tile(UB * np.ones(D), (N, 1))

    # Initialize top predator
    Top_predator_pos = np.zeros(D)
    Top_predator_fit = np.inf

    # Initialize fitness and history
    fitness = np.full(N, np.inf)
    Convergence_curve = np.zeros(Tmax)
    stepsize = np.zeros((N, D))

    # Marine Memory
    Prey_old = Prey.copy()
    fit_old = fitness.copy()

    for Iter in range(Tmax):
        # Detecting top predator
        for i in range(N):
            # Boundary checking
            Flag4ub = Prey[i, :] > UB
            Flag4lb = Prey[i, :] < LB
            Prey[i, :] = (Prey[i, :] * (~(Flag4ub | Flag4lb)) +
                         UB * Flag4ub + LB * Flag4lb)

            # Fitness evaluation
            fitness[i] = evaluate(func, Prey[i:i+1, :])[0]

            # Update top predator
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :].copy()

        # Marine Memory saving
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = Prey.copy()

        Inx = fit_old < fitness
        Indx = np.tile(Inx[:, np.newaxis], (1, D))
        Prey = Indx * Prey_old + ~Indx * Prey
        fitness = Inx * fit_old + ~Inx * fitness

        fit_old = fitness.copy()
        Prey_old = Prey.copy()

        # Elite matrix (Eq. 10)
        Elite = np.tile(Top_predator_pos, (N, 1))

        # CF calculation
        CF = (1 - Iter / Tmax) ** (2 * Iter / Tmax)

        # Levy and Brownian random vectors
        RL = 0.05 * levy_flight_vector(N, D, 1.5)
        RB = rng.standard_normal((N, D))

        # Update prey positions based on phase
        for i in range(N):
            for j in range(D):
                R = rng.random()

                # Phase 1: High velocity ratio (Eq. 12)
                if Iter < Tmax / 3:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]

                # Phase 2: Unit velocity ratio (Eqs. 13 & 14)
                elif Iter < 2 * Tmax / 3:
                    if i > N / 2:
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]

                # Phase 3: Low velocity ratio (Eq. 15)
                else:
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]

        # Detecting top predator (after movement)
        for i in range(N):
            # Boundary checking
            Flag4ub = Prey[i, :] > UB
            Flag4lb = Prey[i, :] < LB
            Prey[i, :] = (Prey[i, :] * (~(Flag4ub | Flag4lb)) +
                         UB * Flag4ub + LB * Flag4lb)

            # Fitness evaluation
            fitness[i] = evaluate(func, Prey[i:i+1, :])[0]

            # Update top predator
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :].copy()

        # Marine Memory saving (second time)
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = Prey.copy()

        Inx = fit_old < fitness
        Indx = np.tile(Inx[:, np.newaxis], (1, D))
        Prey = Indx * Prey_old + ~Indx * Prey
        fitness = Inx * fit_old + ~Inx * fitness

        fit_old = fitness.copy()
        Prey_old = Prey.copy()

        # Eddy formation and FADs effect (Eq. 16)
        if rng.random() < FADs:
            U = rng.random((N, D)) < FADs
            Prey = Prey + CF * ((Xmin + rng.random((N, D)) * (Xmax - Xmin)) * U)
        else:
            r = rng.random()
            Rs = N
            # Random permutations for FADs effect
            idx1 = rng.permutation(Rs)
            idx2 = rng.permutation(Rs)
            stepsize = (FADs * (1 - r) + r) * (Prey[idx1, :] - Prey[idx2, :])
            Prey = Prey + stepsize

        # Save convergence
        Convergence_curve[Iter] = Top_predator_fit
        
        # Callback for real-time updates
        if callback:
            callback(Iter + 1, Top_predator_fit)

    return Top_predator_pos, Top_predator_fit, Convergence_curve
