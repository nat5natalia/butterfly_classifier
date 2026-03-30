from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import CFG
from src.utils import set_seed


def stratified_group_split(
    df: pd.DataFrame,
    test_size: float = CFG.test_size,
    val_size: float = CFG.val_size,
    group_col: str = CFG.group_col,
    random_state: int = CFG.seed,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets ensuring
    that all rows from the same group (e.g., observation_id)
    stay together.
    """
    set_seed(random_state)

    # First split: separate test set
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Second split: separate validation from remaining
    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df[group_col]))
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    return train_df, val_df, test_df