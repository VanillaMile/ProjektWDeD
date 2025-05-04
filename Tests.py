import polars as pl
import numpy as np
from example_data import *
from time import time

class StopException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

class Tests:
    def check_data_size(self) -> None:
        """
        Sprawdza, czy liczba obiektów w danych oryginalnych i zdyskretyzowanych jest taka sama.
        W przypadku niezgodności zgłasza wyjątek StopException.
        """
        original_size = self.data.shape[0]
        discretized_size = self.disc_data.shape[0]
        test_name = 'Sprawdzenie liczby obiektów'

        if original_size != discretized_size:
            self._test_passed(test_name, False)
            raise StopException(
                f"Niezgodna liczba obiektów! Oryginalne: {original_size}, Zdyskretyzowane: {discretized_size}"
            )
        else:
            self._test_passed(test_name, True)
            print(f"Liczba obiektów zgodna: {original_size}")
    def __init__(self, data_path: str, disc_data_path: str, has_header: bool) -> None:
        self.data = pl.read_csv(data_path, separator=',', has_header=has_header)
        self.disc_data = pl.read_csv(disc_data_path, separator=',', has_header=has_header)

        if not has_header:
            labels = ['x' + str(i) for i in range(self.data.shape[1] - 1)]
            labels.append('Dec')
            self.data.columns = labels
            self.disc_data.columns = labels
        
    def check_data(self) -> None:
        print(self.data)
        print(self.disc_data)

        print(self.data['Dec'])

    def test_all(self) -> None:
        self.compare_non_deterministic_objects(lossless=True)

    def _test_passed(self, message: str, passed: bool) -> None:
        green = '\033[0;32m'
        red = '\033[0;31m'
        if passed:
            print(f'{message.ljust(70, ".")}{green} passed')
        else:
            print(f'{message.ljust(70, ".")}{red} FAILED')
        # clear
        print("\033[0;0m", end='')

    def _get_non_deterministic_objects_original_with_loss(self) -> int:
        """Returns number of unique non-deterministic objects in the data"""
        data = self.data.unique()
        non_deterministic_obj = data.select(data.columns[:-1]).is_duplicated().sum()
        return non_deterministic_obj
    
    def _get_non_deterministic_objects_disc_with_loss(self) -> int:
        """Returns number of unique non-deterministic ranges in the data"""
        data = self.disc_data.unique()
        non_deterministic_rng = data.select(data.columns[:-1]).is_duplicated().sum()
        return non_deterministic_rng
    
    
    def _get_non_deterministic_objects_original_lossless(self, how: str = 'inner') -> int:
        """Gets a list of all unique non-deterministic objects in the data and than returns how many of them are in source data.
        For how use 'inner' or 'semi'"""
        temp_data = self.data.unique()
        odd_ones = temp_data.filter(temp_data.select(temp_data.columns[:-1]).is_duplicated())
        joined = self.data.join(odd_ones, on=self.data.columns, how=how)
        return joined.shape[0]

    def _get_non_deterministic_objects_disc_lossless(self, how: str = 'inner') -> int:
        """Gets a list of all unique non-deterministic ranges in the data and than returns how many of them are in source data.
        For how use 'inner' or 'semi'"""
        temp_data = self.disc_data.unique()
        odd_ones = temp_data.filter(temp_data.select(temp_data.columns[:-1]).is_duplicated())
        joined = self.disc_data.join(odd_ones, on=self.disc_data.columns, how=how)
        return joined.shape[0]

    def compare_non_deterministic_objects(self, lossless: bool = True, debug: bool = False) -> None:
        """
        with_loss method is skipping information which in rare cases may give wrong results.

        If non-deterministic ranges are being merged into one range in discretized data
        that means 2 pairs of non-deterministic objects like [1,1,1], [1,1,2], [2,2,1] and [2,2,2] 
        can be merged into 1 range like [0;3,0;3,1] and [0;3,0;3,2].

        In that case _get_non_deterministic_objects_original_with_loss() will return 4
        and _get_non_deterministic_objects_disc_with_loss() will return 2.

        So simply checking if the numbers of non-deterministic objects in original data 
        and discretized data are the same will not work.

        This opens a possibility of having bad non-deterministic ranges in discretized data passed instead of good ones.

        More decisions doesn't affect the result since having 3+ decisions at same spot will result in having 3 unique ranges in discretized data.

        with_loss DOESN'T RETURN NUMBER OF NON-DETERMINISTIC OBJECTS/RANGES IN DATA.
        It only returns number of unique non-deterministic objects/ranges in data.

        It is however, twice as fast as the lossless version, so if the problem of bad ranges is fixed in other tests
        this may by a good way to check for non-deterministic objects/ranges in data.
        """
        test_name = 'Compare non-deterministic objects'
        if lossless:
            if self._get_non_deterministic_objects_disc_lossless() != self._get_non_deterministic_objects_original_lossless():
                self._test_passed(test_name, False)
                raise StopException(f"""Number of non-deterministic objects is not the same as in original data.
                In original data: {self._get_non_deterministic_objects_original_lossless()} non-deterministic objects 
                In discretized data: {self._get_non_deterministic_objects_disc_lossless()} non-deterministic objects""")
        else:
            if self._get_non_deterministic_objects_disc_with_loss() > self._get_non_deterministic_objects_original_with_loss():
                self._test_passed(test_name, False)
                raise StopException(f"""Number of non-deterministic objects is greater than original data.
                In original data: {self._get_non_deterministic_objects_original_with_loss()} unique non-deterministic objects
                In discretized data: {self._get_non_deterministic_objects_disc_with_loss()} unique non-deterministic objects""")
            
        self._test_passed(test_name, True)
        
        if debug:
                print(f'In original data with loss: {self._get_non_deterministic_objects_original_with_loss()} unique non-deterministic objects')
                print(f'In discretized data with loss: {self._get_non_deterministic_objects_disc_with_loss()} unique non-deterministic ranges')
                print(f'In original data lossless: {self._get_non_deterministic_objects_original_lossless()} non-deterministic objects')
                print(f'In discretized data lossless: {self._get_non_deterministic_objects_disc_lossless()} non-deterministic objects')

    # Other methods here

# Test the testing algorithm
def test_compare_non_deterministic_objects(debug: bool = True, plot: bool = False) -> None:
    color = '\033[0;33m'
    clear = '\033[0;0m'

    print (f'{color}Testing Iris 2D non-deterministic data with good discretization{clear}')
    irisND = Iris2DNonDeterministic()
    test_irisND = Tests(data_path=irisND.data_path, disc_data_path=irisND.disc_path, has_header=False)
    test_irisND.compare_non_deterministic_objects(lossless=True, debug=debug)
    if plot: 
        irisND.plot()

    try:
        print (f'{color}Testing Iris 2D non-deterministic data with bad discretization{clear}')
        irisNDBAD = BADIris2DNonDeterministic()
        test_irisNDBAD = Tests(data_path=irisNDBAD.data_path, disc_data_path=irisNDBAD.disc_path, has_header=False)
        test_irisNDBAD.compare_non_deterministic_objects(lossless=True, debug=debug)
    except StopException as e:
        # purple = '\033[0;35m'
        print('\033[0;35m', end='')
        print(e)
        print("Test failed successfully")
        if plot:
            irisNDBAD.plot()
        # clear
        print("\033[0;0m", end='')

    print (f'{color}Testing Iris 3D non-deterministic data with good discretization{clear}')
    iris3D = Iris3D()
    test_iris3D = Tests(data_path=iris3D.data_path, disc_data_path=iris3D.disc_path, has_header=False)
    test_iris3D.compare_non_deterministic_objects(lossless=True, debug=debug)
    if plot:
        iris3D.plot()

    try:
        print (f'{color}Testing Iris 3D non-deterministic data with bad discretization{clear}')
        iris3DBAD = Iris3DBAD()
        test_iris3DBAD = Tests(data_path=iris3DBAD.data_path, disc_data_path=iris3DBAD.disc_path, has_header=False)
        test_iris3DBAD.compare_non_deterministic_objects(lossless=True, debug=debug)
    except StopException as e:
        # purple
        print('\033[0;35m', end='')
        print(e)
        print("Test failed successfully")
        if plot:
            iris3DBAD.plot()
        # clear
        print("\033[0;0m", end='')


def validate_discretization_intervals(self) -> None:
    """
    Sprawdza, czy każda oryginalna wartość atrybutu warunkowego należy do 
    odpowiadającego jej przedziału w danych zdyskretyzowanych.
    W przypadku niezgodności zgłasza wyjątek StopException.
    """
    test_name = 'LA: Sprawdzenie poprawności przypisania do przedziałów'
    print(f"\n{test_name}...")

    conditional_attributes = self.data.columns[:-1] 
    num_rows = self.data.shape[0]

    def check_value_in_interval(value, interval_str):
        interval_str = str(interval_str).strip()
        import re
        import math

       
        try:
            interval_val = float(interval_str)
            return math.isclose(value, interval_val)
        except ValueError:
             pass 

        m = re.match(r"([(\[])\s*(-inf|[-+]?\d*\.?\d+)\s*;\s*(inf|[-+]?\d*\.?\d+)\s*([)\]])", interval_str)
        if not m:
            print(f"  OSTRZEŻENIE: Nie można sparsować przedziału: '{interval_str}'")
            return False 

        left_bracket, lower_str, upper_str, right_bracket = m.groups()

        lower = -float('inf') if lower_str == '-inf' else float(lower_str)
        upper = float('inf') if upper_str == 'inf' else float(upper_str)

        lower_ok = value > lower if left_bracket == '(' else value >= lower
        upper_ok = value < upper if right_bracket == ')' else value <= upper

        return lower_ok and upper_ok

    for i in range(num_rows):
        for j, col_name in enumerate(conditional_attributes):
            original_value = self.data[i, j]
            interval_representation = self.disc_data[i, j]

        
            try:
               original_value_float = float(original_value)
            except (ValueError, TypeError):
               continue 

            if not check_value_in_interval(original_value_float, interval_representation):
                self._test_passed(test_name, False)
                raise StopException(
                    f"Błąd dyskretyzacji! Wartość {original_value_float} (wiersz {i}, kolumna '{col_name}') "
                    f"nie należy do przedziału '{interval_representation}'."
                )
        if (i + 1) % 100 == 0: print(f"  Sprawdzono {i+1}/{num_rows} wierszy.")

    self._test_passed(test_name, True)
    print("Wszystkie wartości mieszczą się w swoich przedziałach.")

if __name__ == "__main__":
    test_compare_non_deterministic_objects(debug=True, plot=False)
    # iris3D = Iris3D()
    # test_iris3DBAD = Tests(data_path=iris3D.data_path, disc_data_path=iris3D.disc_path, has_header=False)
    # test_iris3DBAD.test_all()

    