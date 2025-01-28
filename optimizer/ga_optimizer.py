import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 초기 개체군 초기화
def initialize_population(pop_size, hyperparameters):
    population = []
    for _ in range(pop_size):
        chromosome = [round(np.random.choice(hyperparam), 6) for hyperparam in hyperparameters]
        population.append(chromosome)
    return population

# 병렬 적합도 평가
def evaluate_fitness_parallel(args, train_func, population, hyperparameter_names):
    fitness_scores = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_fitness, args, train_func, individual, hyperparameter_names): individual for individual in population}
        for future in as_completed(futures):
            fitness_scores.append(future.result())
    return fitness_scores

# 적합도 함수 평가
def evaluate_fitness(args, train_func, chromosome, hyperparameter_names):
    for i, hyper_name in enumerate(hyperparameter_names):
        setattr(args, hyper_name, round(chromosome[i], 6)) 
    # 여기서 훈련 루프를 실행하고 F1 점수를 반환합니다
    f1 = train_func(args)  # F1 점수 반환
    return f1

# 선택 함수 (적합도 기반 비례 선택)
def selection(population, fitness_scores):
    fitness_probs = fitness_scores / np.sum(fitness_scores)  # 확률로 변환
    selected_indices = np.random.choice(len(population), size=len(population), p=fitness_probs, replace=True)
    return [population[i] for i in selected_indices]

# 교차 함수
def crossover(parent1, parent2, GA_parameter):
    if np.random.rand() < GA_parameter["crossover_rate"]:
        cross_point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:cross_point] + parent2[cross_point:]
        child2 = parent2[:cross_point] + parent1[cross_point:]
        return [child1, child2]
    else:
        return [parent1, parent2]

# 변이 함수
def mutate(chromosome, GA_parameter, hyperparameters):
    if np.random.rand() < GA_parameter["mutation_rate"]:
        mutation_idx = np.random.randint(len(chromosome))
        chromosome[mutation_idx] = round(np.random.choice(hyperparameters[mutation_idx]), 6)
    return chromosome

# GA 최적화 함수
def GA_optimizer(args, GA_parameter, train_func, hyperparameters, hyperparameter_names):
    # 메인 GA 루프
    population = initialize_population(GA_parameter["population_size"], hyperparameters)
    best_f1 = -np.inf
    best_chromosome = None

    for gen in range(GA_parameter["generation"]):
        # 병렬로 적합도 계산
        fitness_scores = np.array(evaluate_fitness_parallel(args, train_func, population, hyperparameter_names))
        
        # F1 점수 출력 (소수점 4자리)
        print(f"Generation {gen}: Best F1 Score = {np.max(fitness_scores):.4f}")

        # 최적의 솔루션 업데이트
        max_f1 = np.max(fitness_scores)
        if max_f1 > best_f1:
            best_f1 = max_f1
            best_chromosome = population[np.argmax(fitness_scores)]

        # 선택 및 교배
        selected_population = selection(population, fitness_scores)
        next_generation = []

        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[min(i + 1, len(selected_population) - 1)]
            children = crossover(parent1, parent2, GA_parameter)
            next_generation.extend([mutate(child, GA_parameter, hyperparameters) for child in children])

        # 모든 염색체를 소수점 6자리로 반올림
        population = [list(map(lambda x: round(x, 6), individual)) for individual in next_generation[:GA_parameter['population_size']]]
        
        # Population 출력 (선택적)
        print(f"Generation {gen} Population: {population}")

    return best_f1, best_chromosome
