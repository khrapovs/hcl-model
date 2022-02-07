from typing import List

import pandas as pd


def get_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """Get a list of duplicate columns.

    https://thispointer.com/how-to-find-drop-duplicate-columns-in-a-dataframe-python-pandas/

    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.

    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    """
    duplicate_column_names = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            other_col = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])

    return list(duplicate_column_names)
