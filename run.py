import numpy as np
from visualize import Visualize
from example_data import example_algorithm, load_data

if __name__ == "__main__":
    data, target = load_data("w_data")
    print("Data shape: ", data.shape)
    print("Target shape: ", target.shape)
    print("Type data: ", type(data))
    print("Type target: ", type(target))
    print("Data[0] types: ", type(data[0][0]), type(data[0][1]))
    print("Target[0] type: ", type(target[0]))
    
    visualizer = Visualize(data=data, target=target, title="W data", labels=['a', 'b'])
    visualizer.visualize(type='2D')
    
