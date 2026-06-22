import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def cyclical_encode(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.array(data, dtype=np.float64)
    sin_enc = np.sin(2 * np.pi * data / period)
    cos_enc = np.cos(2 * np.pi * data / period)
    return sin_enc, cos_enc


def add_time_features(df: pd.DataFrame, datetime_col: str = 'datetime', holidays: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    dt = df[datetime_col]
    
    hour_sin, hour_cos = cyclical_encode(dt.dt.hour.values, 24)
    month_sin, month_cos = cyclical_encode(dt.dt.month.values, 12)
    
    df['hour_sin'] = hour_sin
    df['hour_cos'] = hour_cos
    df['month_sin'] = month_sin
    df['month_cos'] = month_cos
    
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    
    df['day_type'] = 1
    df.loc[df['is_weekend'] == 1, 'day_type'] = 2
    
    if holidays is not None:
        holiday_dates = pd.to_datetime(holidays).date
        df['is_holiday'] = dt.dt.date.isin(holiday_dates).astype(int)
        df.loc[df['is_holiday'] == 1, 'day_type'] = 3
    else:
        df['is_holiday'] = 0
    
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, datetime_col: str = 'datetime', lags: Optional[List[int]] = None, freq: str = '15T') -> pd.DataFrame:
    if lags is None:
        if freq == '15T':
            lags = [96, 672, 1344]
        elif freq == 'H':
            lags = [24, 168, 336]
        else:
            lags = [24, 168, 336]
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    df = df.reset_index()
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, datetime_col: str = 'datetime', windows: Optional[List[int]] = None, freq: str = '15T') -> pd.DataFrame:
    if windows is None:
        if freq == '15T':
            windows = [16, 48, 96]
        elif freq == 'H':
            windows = [4, 12, 24]
        else:
            windows = [4, 12, 24]
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
    
    df = df.reset_index()
    return df


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    df = df.copy()
    weather_df = weather_df.copy()
    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    weather_df[datetime_col] = pd.to_datetime(weather_df[datetime_col])
    
    weather_cols = [col for col in weather_df.columns if col != datetime_col]
    df = df.merge(weather_df, on=datetime_col, how='left')
    
    for col in weather_cols:
        if f'{col}_forecast' in weather_df.columns:
            df[f'{col}_forecast'] = weather_df[f'{col}_forecast']
    
    return df


def select_features_correlation(df: pd.DataFrame, target_col: str, threshold: float = 0.1, exclude_cols: Optional[List[str]] = None) -> List[str]:
    if exclude_cols is None:
        exclude_cols = ['datetime']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
    
    corr_matrix = df[feature_cols + [target_col]].corr()
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
    
    selected = target_corr[target_corr >= threshold].index.tolist()
    selected = [col for col in selected if col != target_col]
    
    return selected


def select_features_importance(df: pd.DataFrame, target_col: str, k: int = 20, method: str = 'f_regression', exclude_cols: Optional[List[str]] = None, random_state: int = 42) -> List[str]:
    if exclude_cols is None:
        exclude_cols = ['datetime']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
    
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(feature_cols)))
    elif method == 'random_forest':
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=feature_cols)
        selected = importance.sort_values(ascending=False).head(k).index.tolist()
        return selected
    else:
        raise ValueError(f"Unknown method: {method}. Use 'f_regression', 'mutual_info', or 'random_forest'.")
    
    selector.fit(X, y)
    selected = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    return selected


def build_features(df: pd.DataFrame, target_col: str, weather_df: Optional[pd.DataFrame] = None, datetime_col: str = 'datetime', freq: str = '15T', holidays: Optional[List[str]] = None, lags: Optional[List[int]] = None, rolling_windows: Optional[List[int]] = None) -> pd.DataFrame:
    df = df.copy()
    
    df = add_time_features(df, datetime_col=datetime_col, holidays=holidays)
    
    df = add_lag_features(df, target_col=target_col, datetime_col=datetime_col, lags=lags, freq=freq)
    
    df = add_rolling_features(df, target_col=target_col, datetime_col=datetime_col, windows=rolling_windows, freq=freq)
    
    if weather_df is not None:
        df = add_weather_features(df, weather_df, datetime_col=datetime_col)
    
    return df


def standardize_features(df: pd.DataFrame, feature_cols: List[str], mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[pd.DataFrame, float, float]:
    df = df.copy()
    data = df[feature_cols].values.astype(np.float64)
    
    if mean is None or std is None:
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std = np.where(std == 0, 1.0, std)
    
    standardized = (data - mean) / std
    df[feature_cols] = standardized
    
    return df, mean, std


def normalize_features(df: pd.DataFrame, feature_cols: List[str], min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[pd.DataFrame, float, float]:
    df = df.copy()
    data = df[feature_cols].values.astype(np.float64)
    
    if min_val is None or max_val is None:
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
    else:
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
    
    normalized = (data - min_val) / range_val
    df[feature_cols] = normalized
    
    return df, min_val, max_val


def inverse_standardize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.array(data) * std + mean


def inverse_normalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    range_val = max_val - min_val
    if range_val == 0:
        return np.array(data) + min_val
    return np.array(data) * range_val + min_val
