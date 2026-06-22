import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def generate_time_range(start_date: str, end_date: str, freq: str = '15min') -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def check_time_continuous(datetimes: pd.DatetimeIndex, expected_freq: str = '15min') -> Tuple[bool, List[Tuple[datetime, datetime]]]:
    gaps = []
    expected_delta = pd.Timedelta(expected_freq)
    
    for i in range(1, len(datetimes)):
        actual_delta = datetimes[i] - datetimes[i-1]
        if actual_delta != expected_delta:
            gaps.append((datetimes[i-1], datetimes[i]))
    
    return (len(gaps) == 0, gaps)


def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    data = np.array(data)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - k * iqr
    upper_bound = q75 + k * iqr
    return (data < lower_bound) | (data > upper_bound)


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    data = np.array(data, dtype=np.float64)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data), mean, 1.0
    return (data - mean) / std, mean, std


def denormalize_data(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.array(data) * std + mean


def min_max_scale(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    data = np.array(data, dtype=np.float64)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data), min_val, max_val
    return (data - min_val) / (max_val - min_val), min_val, max_val


def inverse_min_max_scale(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return np.array(data) * (max_val - min_val) + min_val


def sliding_window(data: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
    data = np.array(data)
    n_samples = len(data)
    if window_size > n_samples:
        return np.array([])
    indices = np.arange(window_size)[None, :] + step_size * np.arange((n_samples - window_size) // step_size + 1)[:, None]
    return data[indices]


def create_sequences(features: np.ndarray, target: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)


def datetime_features(dt: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=dt)
    df['hour'] = dt.hour
    df['day_of_week'] = dt.dayofweek
    df['day_of_month'] = dt.day
    df['month'] = dt.month
    df['is_weekend'] = (dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = dt.is_month_start.astype(int)
    df['is_month_end'] = dt.is_month_end.astype(int)
    df['quarter'] = dt.quarter
    return df


def cyclical_encoding(data: pd.Series, period: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.array(data)
    sin_enc = np.sin(2 * np.pi * data / period)
    cos_enc = np.cos(2 * np.pi * data / period)
    return sin_enc, cos_enc


def time_series_split(data: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    return train, val, test


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def format_number(num: float, decimals: int = 2) -> str:
    return f"{num:.{decimals}f}"


def format_percentage(num: float, decimals: int = 2) -> str:
    return f"{num:.{decimals}f}%"


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_season(month: int) -> str:
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    else:
        return '冬季'


def aggregate_to_hourly(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    hourly = df.resample('H').mean()
    hourly.reset_index(inplace=True)
    return hourly


def aggregate_to_daily(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    daily = df.resample('D').agg({
        col: ['mean', 'min', 'max', 'sum'] for col in df.columns
    })
    daily.columns = [f"{col}_{agg}" for col, agg in daily.columns]
    daily.reset_index(inplace=True)
    return daily


def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = df1.copy()
    df2 = df2.copy()
    df1[key] = pd.to_datetime(df1[key])
    df2[key] = pd.to_datetime(df2[key])
    common_keys = set(df1[key]).intersection(set(df2[key]))
    df1 = df1[df1[key].isin(common_keys)].sort_values(key).reset_index(drop=True)
    df2 = df2[df2[key].isin(common_keys)].sort_values(key).reset_index(drop=True)
    return df1, df2


def max_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.max(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def get_prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = y_pred - y_true
    mean_error = float(np.mean(errors))
    if mean_error > 0:
        return "偏高为主"
    elif mean_error < 0:
        return "偏低为主"
    else:
        return "无明显偏移"


def get_time_period(hour: int) -> str:
    if 0 <= hour < 7:
        return "谷时(0-7点)"
    elif 7 <= hour < 11 or 15 <= hour < 19:
        return "平时(7-11,15-19点)"
    elif 11 <= hour < 15 or 19 <= hour < 22:
        return "峰时(11-15,19-22点)"
    elif 22 <= hour < 24:
        return "深谷(22-24点)"
    else:
        return "其他"


def calculate_period_mape(result_df: pd.DataFrame, datetime_col: str = 'datetime', 
                          actual_col: str = 'actual', pred_col: str = 'predicted') -> Dict[str, float]:
    df = result_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['period'] = df['hour'].apply(get_time_period)
    
    period_mapes = {}
    periods = ["谷时(0-7点)", "平时(7-11,15-19点)", "峰时(11-15,19-22点)", "深谷(22-24点)"]
    
    for period in periods:
        period_data = df[df['period'] == period]
        if len(period_data) > 0:
            period_mapes[period] = mape(period_data[actual_col].values, period_data[pred_col].values)
        else:
            period_mapes[period] = 0.0
    
    return period_mapes


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        'mape': mape(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'max_ape': max_absolute_percentage_error(y_true, y_pred),
        'bias': get_prediction_bias(y_true, y_pred)
    }


def recommend_models(lgbm_period_mapes: Dict[str, float], 
                     lstm_period_mapes: Dict[str, float]) -> Tuple[Dict[str, str], str]:
    recommendations = {}
    periods = ["谷时(0-7点)", "平时(7-11,15-19点)", "峰时(11-15,19-22点)", "深谷(22-24点)"]
    period_short = ["谷时", "平时", "峰时", "深谷"]
    
    for period, short in zip(periods, period_short):
        if lgbm_period_mapes[period] < lstm_period_mapes[period]:
            recommendations[period] = "LightGBM"
        else:
            recommendations[period] = "LSTM"
    
    lgbm_better = [p for p, m in recommendations.items() if m == "LightGBM"]
    lstm_better = [p for p, m in recommendations.items() if m == "LSTM"]
    
    if len(lstm_better) == 0:
        summary = "各时段均建议使用LightGBM模型"
    elif len(lgbm_better) == 0:
        summary = "各时段均建议使用LSTM模型"
    else:
        lgbm_short = [period_short[periods.index(p)] for p in lgbm_better]
        lstm_short = [period_short[periods.index(p)] for p in lstm_better]
        
        if len(lstm_better) == 1:
            summary = f"{lstm_short[0]}建议用LSTM，其余时段LightGBM表现更优"
        elif len(lgbm_better) == 1:
            summary = f"{lgbm_short[0]}建议用LightGBM，其余时段LSTM表现更优"
        else:
            lstm_str = "、".join(lstm_short)
            lgbm_str = "、".join(lgbm_better)
            summary = f"{lstm_str}建议用LSTM，{lgbm_str}建议用LightGBM"
    
    return recommendations, summary
