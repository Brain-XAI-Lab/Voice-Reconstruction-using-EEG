import numpy as np
# Random Search 함수
def random_search(args, train_func, hyperparameters, hyperparameter_names, num_samples):
    best_f1 = -np.inf
    best_chromosome = None
    
    for _ in range(num_samples):
        # 무작위 하이퍼파라미터 조합 생성
        chromosome = [round(np.random.choice(hyperparam), 6) for hyperparam in hyperparameters]
        
        # 하이퍼파라미터를 args에 설정
        for i, hyper_name in enumerate(hyperparameter_names):
            setattr(args, hyper_name, chromosome[i])
        
        # 모델 훈련 및 F1 점수 계산
        f1 = train_func(args)
        
        # 최적의 F1 점수와 조합 업데이트
        if f1 > best_f1:
            best_f1 = f1
            best_chromosome = chromosome
    
    return best_f1, best_chromosome
