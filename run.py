import numpy as np
import polars as pl
from Tests import Tests, StopException
from time import time

def example_algorithm(data_path: str) -> None:
    print(f"    -> Symulacja dyskretyzacji dla: {data_path}")
    pass

if __name__ == "__main__":
    data_paths = ['data1.csv', 'data1.csv']
    disc_data_paths = ['DISCdata1.csv', 'DISCdata1.csv']

    purple = '\033[0;35m'
    clear = '\033[0;0m'
    red = '\033[0;31m'
    green = '\033[0;32m'
    yellow = '\033[0;33m'

    results_list = []

    for i, (data_path, disc_data_path) in enumerate(zip(data_paths, disc_data_paths)):
        print(f"\n{yellow}--- Testowanie pary plików i obliczanie Oceny: {data_path} i {disc_data_path} (i={i}) ---{clear}")
        try:
            pair_start_time = time()

            tests = Tests(data_path, disc_data_path, has_header=False)

            tests.test_all()

            cuts_i = tests.count_discretization_cuts()
            det_i = tests.non_deterministic_objects_original
        
            pair_end_time = time()
            time_i = pair_end_time - pair_start_time

            ocena_i = 0.5 * float(det_i) + 0.25 * float(cuts_i) + time_i
            result_tuple = (i, time_i, det_i, cuts_i, ocena_i)
            results_list.append(result_tuple)

            print(f"Liczba niedeterministycznych obiektów (det_{i}): {purple}{det_i}{clear}")
            print(f"Czas walidacji i obliczeń dla pary (time_{i}): {purple}{time_i:.4f} sekund{clear}")
            print(f"Obliczona Ocena_{i}: {purple}{ocena_i:.4f}{clear}")
            print(f"{green}--- Testy i obliczenia dla {data_path} zakończone pomyślnie ---{clear}")
        except FileNotFoundError:
            print(f"{red}BŁĄD:{clear} {purple}Nie znaleziono jednego z plików: {data_path} lub {disc_data_path}{clear}")
        except StopException as e:
            print(f"{red}BŁĄD:{clear} {purple}{e}{clear}")
        except Exception as e:
            print(f"{red}BŁĄD:{clear} {purple}podczas testowania lub obliczeń dla {data_path}: {e}{clear}")

    print(f"\n{purple}========= Zakończono wszystkie operacje ========={clear}")

    df = pl.DataFrame(
        {
            "i": [res[0] for res in results_list],
            "time_i": [res[1] for res in results_list],
            "det_i": [res[2] for res in results_list],
            "cuts_i": [res[3] for res in results_list],
            "Ocena_i": [res[4] for res in results_list],
        }
    )
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_rows(i+1)
    print(df)

    # df.write_csv("results.csv", separator=',')

    last_ocena = results_list[-1][4] if results_list else None
    print(f"\nOstatnia obliczona Ocena: {last_ocena:.4f}" if last_ocena is not None else "\nNie obliczono żadnej Oceny.")
