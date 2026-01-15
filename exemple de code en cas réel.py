import numpy as np

def slime_mould_algorithm(obj_func, dim, lb, ub, pop_size=30, max_iter=100):

    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = []

    for t in range(max_iter):

        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]

        best_current = fitness[0]
        worst_current = fitness[-1]

        weights = np.zeros(pop_size)
        for i in range(pop_size):
            if i < pop_size / 2:
                weights[i] = 1 + np.random.rand() * np.log10(
                    (best_current - fitness[i]) /
                    (best_current - worst_current + 1e-10) + 1
                )
            else:
                weights[i] = 1 - np.random.rand() * np.log10(
                    (best_current - fitness[i]) /
                    (best_current - worst_current + 1e-10) + 1
                )

        a = np.arctanh(1 - t / max_iter)

        for i in range(pop_size):
            r = np.random.rand()
            if r < 0.03:
                population[i] = np.random.uniform(lb, ub, dim)
            else:
                vb = np.random.uniform(-a, a, dim)
                vc = np.random.uniform(-1, 1, dim)

                population[i] = best_position + vb * (
                    weights[i] * population[np.random.randint(pop_size)]
                    - population[np.random.randint(pop_size)]
                ) + vc

            population[i] = np.clip(population[i], lb, ub)

        fitness = np.array([obj_func(ind) for ind in population])

        current_best = np.argmin(fitness)
        if fitness[current_best] < best_fitness:
            best_fitness = fitness[current_best]
            best_position = population[current_best].copy()

        convergence_curve.append(best_fitness)

        print(f"Iteration {t+1}/{max_iter} | Best Energy = {best_fitness:.6f}")

    return best_position, best_fitness, convergence_curve
