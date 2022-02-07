import pandas as pd

from hcl_model.utils.get_duplicate_columns import get_duplicate_columns


def test_get_duplicate_columns() -> None:
    students = [
        ("jack", 34, "Sydeny", 34, "Sydeny", 34),
        ("Riti", 30, "Delhi", 30, "Delhi", 30),
        ("Aadi", 16, "New York", 16, "New York", 16),
        ("Riti", 30, "Delhi", 30, "Delhi", 30),
        ("Riti", 30, "Delhi", 30, "Delhi", 30),
        ("Riti", 30, "Mumbai", 30, "Mumbai", 30),
        ("Aadi", 40, "London", 40, "London", 40),
        ("Sachin", 30, "Delhi", 30, "Delhi", 30),
    ]

    df = pd.DataFrame(students, columns=["Name", "Age", "City", "Marks", "Address", "Pin"])

    duplicate_column_names = get_duplicate_columns(df)

    assert set(duplicate_column_names) == {"Marks", "Address", "Pin"}
