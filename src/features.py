from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

EXCLUDE_COLUMNS = ["season", "date", "game_id", "home_win"]

def select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    numeric_cols, categorical_cols = select_feature_columns(df)

    numeric_transformer = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]) if len(categorical_cols) > 0 else "drop"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    feature_names = numeric_cols + categorical_cols
    return preprocessor, feature_names

