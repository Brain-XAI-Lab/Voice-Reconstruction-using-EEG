import numpy as np

# 초기 개체군 초기화
def initialize_population(pop_size, hyperparameters):
    population = []
    for _ in range(pop_size):
        chromosome = [round(np.random.choice(hyperparam), 6) for hyperparam in hyperparameters]
        population.append(chromosome)
    return population

# PSO 함수
def pso_optimizer(args, train_func, hyperparameters, hyperparameter_names, pso_params):
    num_particles = pso_params["num_particles"]
    num_iterations = pso_params["num_iterations"]
    w = pso_params["inertia_weight"]
    c1 = pso_params["cognitive_coeff"]
    c2 = pso_params["social_coeff"]

    # 초기화
    particles = initialize_population(num_particles, hyperparameters)
    velocities = [[0] * len(hyperparameters) for _ in range(num_particles)]
    personal_best = particles[:]
    personal_best_scores = [-np.inf] * num_particles
    global_best = None
    global_best_score = -np.inf

    for _ in range(num_iterations):
        for i, particle in enumerate(particles):
            # 적합도 계산
            for j, hyper_name in enumerate(hyperparameter_names):
                setattr(args, hyper_name, round(particle[j], 6))
            f1 = train_func(args)

            # 개인 및 글로벌 최적 업데이트
            if f1 > personal_best_scores[i]:
                personal_best[i] = particle
                personal_best_scores[i] = f1
            if f1 > global_best_score:
                global_best = particle
                global_best_score = f1

        # 입자 업데이트
        for i, particle in enumerate(particles):
            for j in range(len(hyperparameters)):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i][j] = (
                    w * velocities[i][j]
                    + c1 * r1 * (personal_best[i][j] - particle[j])
                    + c2 * r2 * (global_best[j] - particle[j])
                )
                particle[j] = round(
                    particle[j] + velocities[i][j], 6
                )  # 위치 업데이트
                # 하이퍼파라미터 범위를 벗어나지 않도록 클리핑
                particle[j] = max(hyperparameters[j][0], min(particle[j], hyperparameters[j][-1]))

    return global_best_score, global_best
