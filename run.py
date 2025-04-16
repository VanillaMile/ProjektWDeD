import numpy as np
import polars as pl

from Tests import Tests

def example_algorithm(data_path: str) -> None:
    pass

if __name__ == "__main__":
    # Implement method to input these paths
    data_paths = ['data1.csv']
    disc_data_paths = ['DISCdata1.csv']

    # Measure time of this
    for path in data_paths:
        example_algorithm(path)
    
    for data_path, disc_data_path in zip(data_paths, disc_data_paths):
        tests = Tests(data_path, disc_data_path)
        tests.check_data()
