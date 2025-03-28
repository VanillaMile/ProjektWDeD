from sklearn.datasets import load_iris
import numpy as np

def example_algorithm(dataset) -> list:
    w_data = [
        #   aDISC;       bDISC;    Dec
        ["0.9; 1.95", "5.35; 6.8", 3],
        ["1.95; 4.2", "5.35; 6.8", 2],
        ["1.95; 4.2", "2.1; 5.35", 1],
        ["0.9; 1.95", "2.1; 5.35", 2],
        ["1.95; 4.2", "5.35; 6.8", 2],
        ["1.95; 4.2", "2.1; 5.35", 1],
        ["0.9; 1.95", "5.35; 6.8", 3]
    ]
    iris_data = []

    if dataset == "w_data" or dataset == "w":
        return w_data
    if dataset == "iris" or dataset == "i":
        return iris_data
    
    return None

def load_data(dataset) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "iris" or dataset == "i":
        iris = load_iris()
        data = iris.data
        target = iris.target
    
    if dataset == "w_data" or dataset == "w":
        data = [
            [0.9, 6.2],
            [2.7, 6.2],
            [2.7, 2.1],
            [1.2, 4.5],
            [3.3, 6.8],
            [4.3, 4.5],
            [1.2, 6.2]
        ]
        target = [3, 2, 1, 2, 2, 1, 3]
        data = np.array(data, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
    
    return data, target