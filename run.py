import numpy as np
import polars as pl

if __name__ == "__main__":
    data = pl.read_csv('data1.csv', separator=',')
    desc_data = pl.read_csv('DISCdata1.csv', separator=',')

    print(data)
    print(type(data))
    print(desc_data)
    print(type(desc_data))

    
