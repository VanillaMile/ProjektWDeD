import numpy as np
import polars as pl
from Tests import Tests, StopException
from time import time
from Porada import start_algorithm as example_algorithm
# from Domanski import Algorithm
# example_algorithm = Algorithm.example_algorithm
# from Jakubowski import greedy_discretization as example_algorithm

if __name__ == "__main__":
    # data_paths = ['data1.csv', 'data1.csv', 'example_data_csv/iris2D.csv', 'example_data_csv/iris3D.csv', 
    #               'example_data_csv/iris3D.csv', 'example_data_csv/nodec.csv', 
    #               'example_data_csv/iris2Dnondeterministic.csv', 'example_data_csv/BADiris2Dnondeterministic.csv']
    # disc_data_paths = ['DISCdata1.csv', 'DISCdata1.csv', 'example_disc_csv/DISCiris2D.csv', 
    #                    'example_disc_csv/DISCiris3D.csv', 'example_disc_csv/DISCiris3DBAD.csv', 'example_disc_csv/DISCnodec.csv',
    #                    'example_disc_csv/DISCiris2Dnondeterministic.csv', 'example_disc_csv/DISCBADiris2Dnondeterministic.csv']

    data_paths = ['data2.csv']
    disc_data_paths = ['DISCdata2.csv']

    purple = '\033[0;35m'
    clear = '\033[0;0m'
    red = '\033[0;31m'
    green = '\033[0;32m'
    yellow = '\033[0;33m'

    results_list = []
    times = []

    for data_path in data_paths:
        try:
            print(f"\n{yellow}---> Mierzenie czasu algorytmu dyskretyzującego dla {data_path} ---{clear}")
            pair_start_time = time()
            example_algorithm(data_path)
            pair_end_time = time()
            time_i = pair_end_time - pair_start_time
            times.append(time_i)
        except Exception as e:
            print(f"{red}BŁĄD:{clear} {purple}podczas mierzenia czasu algorytmu dla {data_path}: {e}{clear}")
            times.append('Error')

    for i, (data_path, disc_data_path) in enumerate(zip(data_paths, disc_data_paths)):
        print(f"\n{yellow}--- Testowanie pary plików i obliczanie Oceny: {data_path} i {disc_data_path} (i={i}) ---{clear}")
        try:
            tests = Tests(data_path, disc_data_path, has_header=False)
            if times[i] == 'Error':
                raise StopException(f"Nie można wykonać testów dla {data_path} z powodu błędu podczas pomiaru czasu algorytmu.")

            tests.test_all(debug=False)

            cuts_i = tests.count_discretization_cuts()
            # det_i = tests.non_deterministic_objects_original
            det_i = tests.get_fair_det_count()
        
            ocena_i = 0.5 * float(det_i) + 0.25 * float(cuts_i) + times[i]
            result_tuple = (i, times[i], det_i, cuts_i, ocena_i)
            results_list.append(result_tuple)

            print(f'Łączna liczba unikalnych cięć: {purple}{cuts_i}{clear}')
            print(f"Liczba unikalnych niedeterministycznych przedziałów (det_{i}): {purple}{det_i}{clear}")
            print(f"Sredni czas pracy algorytmu (time_{i}): {purple}{times[i]:.4f} sekund{clear}")
            print(f"Obliczona Ocena_{i}: {purple}{ocena_i:.4f}{clear}")
            print(f"{green}--- Testy i obliczenia dla {data_path} zakończone pomyślnie ---{clear}")
        except FileNotFoundError:
            print(f"{red}BŁĄD:{clear} {purple}Nie znaleziono jednego z plików: {data_path} lub {disc_data_path}{clear}")
            result_tuple = (i, times[i], 'FileNotFound', 'FileNotFound', 'FileNotFound')
            results_list.append(result_tuple)
        except StopException as e:
            print(f"{red}BŁĄD:{clear} {purple}{e}{clear}")
            result_tuple = (i, times[i], 'Failed', 'Failed', 'Failed')
            results_list.append(result_tuple)
        except Exception as e:
            print(f"{red}BŁĄD:{clear} {purple}podczas testowania lub obliczeń dla {data_path}: {e}{clear}")
            result_tuple = (i, times[i], 'Error', 'Error', 'Error')
            results_list.append(result_tuple)

    print(f"\n{purple}========= Zakończono wszystkie operacje ========={clear}")

    df = pl.DataFrame(
        {
            "i": [res[0] for res in results_list],
            "data_path": [data_paths[res[0]] for res in results_list],
            "avg_time_in_s": [res[1] for res in results_list],
            "det_i": [res[2] for res in results_list],
            "cuts_i": [res[3] for res in results_list],
            "Ocena_i": [res[4] for res in results_list],
        }
    , strict=False)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_rows(i+1)
    print(df)
    if not all(isinstance(value, (int, float)) for value in df['Ocena_i']):
        print(f"{red}Podczas weryfikacji danych wystąpił błąd, ocena końcowa nie może być wyliczona{clear}")
    else:
        print(f"{green}Ocena końcowa: {df['Ocena_i'].sum()} (Mniej = Lepiej){clear}")

    # df.write_csv("results.csv", separator=',')
