import polars as pl
import numpy as np

class Tests:
    def __init__(self, data_path: str, disc_data_path: str) -> None:
        self.data = pl.read_csv(data_path, separator=',', has_header=False)
        self.disc_data = pl.read_csv(disc_data_path, separator=',', has_header=False)

        labels = ['x' + str(i) for i in range(self.data.shape[1] - 1)]
        labels.append('Dec')

        self.data.columns = labels
        self.disc_data.columns = labels
        

    def check_data(self) -> None:
        print(self.data)
        print(self.disc_data)

        print(self.data['Dec'])


    # Other methods here