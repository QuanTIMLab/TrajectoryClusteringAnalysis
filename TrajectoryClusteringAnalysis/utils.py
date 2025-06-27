import numpy as np
import pandas as pd
from numba import njit

@njit
def compute_mode_window(arr, kernel_rows, kernel_cols):
    """
    Apply a modal filter (mode of the neighborhood) to a 2D numpy array.

    Parameters:
    - arr (np.ndarray): The input 2D array.
    - kernel_rows (int): Number of rows in the filter kernel.
    - kernel_cols (int): Number of columns in the filter kernel.

    Returns:
    - np.ndarray: The filtered array using the modal filter.
    """
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    for i in range(n_rows):
        for j in range(n_cols):
            row_min = max(0, i - kernel_rows // 2)
            row_max = min(n_rows, i + kernel_rows // 2 + 1)
            col_min = max(0, j - kernel_cols // 2 + 1)
            col_max = min(n_cols, j + kernel_cols // 2 + 1)

            window = []
            for r in range(row_min, row_max):
                for c in range(col_min, col_max):
                    val = arr[r, c]
                    if not np.isnan(val):
                        window.append(int(val))

            if len(window) == 0:
                result[i, j] = np.nan
            else:
                counts = np.bincount(np.array(window))
                result[i, j] = np.argmax(counts)

    return result

def modal_filter_numba(df_numeriques, kernel_size=(10, 7)):
    """
    Apply a modal filter to a pandas DataFrame using a Numba-accelerated function.

    Parameters:
    - df_numeriques (pd.DataFrame): DataFrame containing numerical sequences.
    - kernel_size (tuple): Tuple of (kernel_rows, kernel_cols) for the filter.

    Returns:
    - pd.DataFrame: Filtered DataFrame with the same shape and index/columns.
    """
    arr = df_numeriques.to_numpy()
    filtered_array = compute_mode_window(arr, *kernel_size)
    return pd.DataFrame(filtered_array, index=df_numeriques.index, columns=df_numeriques.columns)
 