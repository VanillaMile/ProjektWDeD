import numpy as np
import polars as pl
from Tests import Tests
from time import time

def example_algorithm(data_path: str) -> None:
    print(f"    -> Symulacja dyskretyzacji dla: {data_path}")
    pass

if __name__ == "__main__":
    data_paths = ['data1.csv']
    disc_data_paths = ['DISCdata1.csv']

    print("========= Rozpoczęcie dyskretyzacji ==========")

    print("\nRozpoczęcie testów walidacyjnych i obliczania Oceny...")

    results_list = []

    for i, (data_path, disc_data_path) in enumerate(zip(data_paths, disc_data_paths)):
        print(f"\n--- Testowanie pary plików i obliczanie Oceny: {data_path} i {disc_data_path} (i={i}) ---")
        try:
            pair_start_time = time()

            tests = Tests(data_path, disc_data_path, has_header=False)

            tests.check_data_size()
            tests.validate_discretization_intervals()

            cuts_i = tests.count_discretization_cuts()
            det_i = tests._get_non_deterministic_objects_disc_lossless()
            print(f"  Liczba niedeterministycznych obiektów (det_{i}): {det_i}")

            pair_end_time = time()
            time_i = pair_end_time - pair_start_time
            print(f"  Czas walidacji i obliczeń dla pary (time_{i}): {time_i:.4f} sekund")

            ocena_i = 0.5 * float(det_i) + 0.25 * float(cuts_i) + time_i

            result_tuple = (i, time_i, det_i, cuts_i, ocena_i)
            results_list.append(result_tuple)

            print(f"  Obliczona Ocena_{i}: {ocena_i:.4f}")
            print(f"--- Testy i obliczenia dla {data_path} zakończone pomyślnie ---")

        except FileNotFoundError:
            print(f"  BŁĄD: Nie znaleziono jednego z plików: {data_path} lub {disc_data_path}")
        except StopException as e:
            print(f"  BŁĄD WALIDACJI (StopException) podczas testowania {data_path}: {e}")
        except Exception as e:
            print(f"  BŁĄD podczas testowania lub obliczeń dla {data_path}: {e}")

    print("\n========= Zakończono wszystkie operacje =========")

    print("\nZebrane wyniki (i, time_i, det_i, cuts_i, Ocena_i):")
    for res in results_list:
        print(f"  ({res[0]}, {res[1]:.4f}, {res[2]}, {res[3]}, {res[4]:.4f})")

    last_ocena = results_list[-1][4] if results_list else None
    print(f"\nOstatnia obliczona Ocena: {last_ocena:.4f}" if last_ocena is not None else "\nNie obliczono żadnej Oceny.")
