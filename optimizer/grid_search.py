import itertools
import numpy as np

# Grid Search 함수
def grid_search(args, train_func, hyperparameters, hyperparameter_names):
    # 모든 하이퍼파라미터 조합 생성
    grid = list(itertools.product(*hyperparameters))
    
    best_f1 = -np.inf
    best_chromosome = None
    
    for chromosome in grid:
        # 하이퍼파라미터를 args에 설정
        for i, hyper_name in enumerate(hyperparameter_names):
            setattr(args, hyper_name, round(chromosome[i], 6))
        
        # 모델 훈련 및 F1 점수 계산
        f1 = train_func(args)
        
        # 최적의 F1 점수와 조합 업데이트
        if f1 > best_f1:
            best_f1 = f1
            best_chromosome = chromosome
    
    return best_f1, best_chromosome
