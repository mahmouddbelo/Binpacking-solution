import random, time


class GeneticBinPacking:
    def __init__(
        self,
        items,
        bin_capacity,
        population_size=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_generations=100,
    ):
        self.items = items
        self.bin_capacity = bin_capacity
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

    def create_initial_solution(self):
        random.shuffle(self.items)
        bins = []
        for item in self.items:
            for bin in bins:
                if sum(bin) + item <= self.bin_capacity:
                    bin.append(item)
                    break
            else:
                bin = [item]
                bins.append(bin)
        return bins

    def evaluate_bin_configuration(self, bins):
        num_bins = len(bins)
        total_remaining_capacity = sum(self.bin_capacity - sum(bin) for bin in bins)
        load_balance_penalty = max(sum(bin) for bin in bins) - min(
            sum(bin) for bin in bins
        )
        fitness = 1 / (num_bins + 1)
        fitness += total_remaining_capacity / (
            self.bin_capacity * num_bins
        )
        fitness -= load_balance_penalty / self.bin_capacity
        return fitness

    def crossover(self, parent1, parent2):
        items = [item for bin in parent1 for item in bin]
        random.shuffle(items)
        bins = []
        for item in items:
            for bin in bins:
                if sum(bin) + item <= self.bin_capacity:
                    bin.append(item)
                    break
            else:
                bin = [item]
                bins.append(bin)
        return bins

    def mutate(self, solution):
        bin1 = random.choice(solution)
        bin2 = random.choice(solution)
        if bin1 != bin2:
            item1 = random.choice(bin1)
            item2 = random.choice(bin2)
            if (sum(bin1) - item1 + item2 <= self.bin_capacity) and (
                sum(bin2) - item2 + item1 <= self.bin_capacity
            ):
                bin1.remove(item1)
                bin2.remove(item2)
                bin1.append(item2)
                bin2.append(item1)
        return solution

    def select_parents(self, population, fitness):
        total_fitness = sum(fitness)
        if total_fitness <= 0:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
        else:
            normalized_fitness = [f / total_fitness for f in fitness]
            parent1 = random.choices(population, weights=normalized_fitness, k=1)[0]
            parent2 = random.choices(population, weights=normalized_fitness, k=1)[0]
        return parent1, parent2

    def evolve_population(self, population, fitness):
        new_population = []
        while len(new_population) < len(population):
            parent1, parent2 = self.select_parents(population, fitness)
            if random.random() < self.crossover_rate:
                offspring1 = self.crossover(parent1, parent2)
                offspring2 = self.crossover(parent2, parent1)
            else:
                offspring1 = parent1
                offspring2 = parent2
            if random.random() < self.mutation_rate:
                offspring1 = self.mutate(offspring1)
            if random.random() < self.mutation_rate:
                offspring2 = self.mutate(offspring2)
            new_population.append(offspring1)
            new_population.append(offspring2)
        return new_population

    def run_genetic_algorithm(self):
        tic = time.time()
        population = [
            self.create_initial_solution() for _ in range(self.population_size)
        ]
        best_solution = None
        best_fitness = 0

        while self.max_generations:
            fitness = [
                self.evaluate_bin_configuration(solution) for solution in population
            ]
            current_best_solution = population[fitness.index(max(fitness))]
            current_best_fitness = max(fitness)
            if current_best_fitness > best_fitness:
                best_solution = current_best_solution
                best_fitness = current_best_fitness
            if len(best_solution) == 2:
                break
            population = self.evolve_population(population, fitness)
            self.max_generations -= 1

        toc = time.time()
        computation_time = toc - tic

        return best_solution, len(best_solution), best_fitness, computation_time