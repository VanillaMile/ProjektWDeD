import numpy as np
import polars as pl

from Tests import Tests

def example_algorithm(data_path: str) -> None:
    pass

if __name__ == "__main__":
    # Implement method to input these paths
    data_paths = ['data1.csv']
    disc_data_paths = ['DISCdata1.csv']

    data = pl.read_csv(data_paths[0], separator=',')
    disc_data = pl.read_csv(disc_data_paths[0], separator=',')

    print(data)
    print(type(data))
    print(disc_data)
    print(type(disc_data))

    tests = Tests()

    # Measure time of this
    for path in data_paths:
        example_algorithm(path)
    
    for path in disc_data_paths:
        tests.check_data(path)
        
         

    
