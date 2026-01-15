import numpy as np

# ==============================
# Fonction objective (à optimiser)
# Exemple : Sphere function
# Minimum global = 0 quand x = [0, 0, ..., 0]
# ==============================
def objective_function(x):
    return np.sum(x ** 2)


# ==============================
# Slime Mould Algorithm (SMA)
# ==============================
def slime_mould_algorithm(obj_func, dim, lb, ub, pop_size=30, max_iter=100):

    # Initialisation de la population
    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in population])

    # Meilleure solution
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Boucle principale
    for t in range(max_iter):

        # Classement des solutions
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        # Calcul des poids
        worst_fitness = fitness[-1]
        best_fitness_current = fitness[0]
        weights = np.zeros(pop_size)

        for i in range(pop_size):
            if i < pop_size / 2:
                weights[i] = 1 + np.random.rand() * np.log10(
                    (best_fitness_current - fitness[i]) /
                    (best_fitness_current - worst_fitness + 1e-10) + 1
                )
            else:
                weights[i] = 1 - np.random.rand() * np.log10(
                    (best_fitness_current - fitness[i]) /
                    (best_fitness_current - worst_fitness + 1e-10) + 1
                )

        # Paramètre adaptatif
        a = np.arctanh(-(t / max_iter) + 1)

        # Mise à jour des positions
        for i in range(pop_size):
            r = np.random.rand()

            if r < 0.03:
                # Exploration aléatoire
                population[i] = np.random.uniform(lb, ub, dim)
            else:
                vb = np.random.uniform(-a, a, dim)
                vc = np.random.uniform(-1, 1, dim)

                population[i] = best_position + vb * (
                    weights[i] * population[np.random.randint(pop_size)]
                    - population[np.random.randint(pop_size)]
                ) + vc

            # Gestion des bornes
            population[i] = np.clip(population[i], lb, ub)

        # Mise à jour du fitness
        fitness = np.array([obj_func(ind) for ind in population])

        # Mise à jour de la meilleure solution globale
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_position = population[current_best_idx].copy()

        # Affichage (optionnel)
        print(f"Iteration {t+1}/{max_iter} - Best Fitness: {best_fitness}")

    return best_position, best_fitness


# ==============================
# Exécution de l'algorithme
# ==============================
if __name__ == "__main__":

    DIMENSION = 5
    LOWER_BOUND = -10
    UPPER_BOUND = 10
    POPULATION_SIZE = 30
    MAX_ITERATIONS = 100

    best_pos, best_fit = slime_mould_algorithm(
        objective_function,
        DIMENSION,
        LOWER_BOUND,
        UPPER_BOUND,
        POPULATION_SIZE,
        MAX_ITERATIONS
    )

    print("\nBest solution found:")
    print("Position:", best_pos)
    print("Fitness:", best_fit)
