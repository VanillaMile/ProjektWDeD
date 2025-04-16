import numpy as np
import polars as pl
from example_data import iris2D, iris3D, iris3DBAD, NoDecisionRange
from Tests import Tests

def example_algorithm(data_path: str) -> None:
    pass

if __name__ == "__main__":
    # Implement method to input these paths
    data_paths = ['data1.csv', 'iris2D.csv', 'iris3D.csv', 'iris3D.csv', 'nodec.csv']
    disc_data_paths = ['DISCdata1.csv', 'DISCiris2D.csv', 'DISCiris3D.csv', 'DISCiris3DBAD.csv', 'DISCnodec.csv']

    # Measure time of this
    for path in data_paths:
        example_algorithm(path)
    
    for data_path, disc_data_path in zip(data_paths, disc_data_paths):
        tests = Tests(data_path, disc_data_path, has_header=False)
        tests.check_data()

    # iris = iris2D()
    # iris.plot()
    iris = iris3D()
    iris.plot()
    # iris = iris3DBAD()
    # iris.plot()
    # nodec = NoDecisionRange()
    # nodec.plot()
