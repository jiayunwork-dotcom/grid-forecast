import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta

from .utils import check_time_continuous, detect_outliers_iqr


def calculate_missing_rate(df: pd.DataFrame) -> pd.DataFrame:
    """按字段统计缺失率。

    Args:
        df: 输入的DataFrame。

    Returns:
        包含缺失统计的DataFrame，列包括：字段名、缺失数量、缺失率。
    """
    missing_stats = pd.DataFrame({
        '字段名': df.columns,
        '缺失数量': df.isna().sum().values,
        '缺失率': df.isna().mean().values * 100
    })
    missing_stats = missing_stats.sort_values('缺失率', ascending=False).reset_index(drop=True)
    return missing_stats


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    k: float = 1.5
) -> Dict[str, Dict[str, Any]]:
    """使用IQR方法检测异常值。

    Args:
        df: 输入的DataFrame。
        columns: 需要检测的列名列表，为None时检测所有数值列。
        k: IQR倍数，默认为1.5。

    Returns:
        异常值检测结果字典，键为列名，值为包含统计信息的字典。
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_result: Dict[str, Dict[str, Any]] = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            continue
        
        outlier_mask = detect_outliers_iqr(valid_data.values, k=k)
        outlier_count = int(np.sum(outlier_mask))
        outlier_indices = valid_data.index[outlier_mask].tolist()
        
        q25, q75 = np.percentile(valid_data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - k * iqr
        upper_bound = q75 + k * iqr
        
        outliers_result[col] = {
            '异常值数量': outlier_count,
            '异常值比例': outlier_count / len(valid_data) * 100,
            'Q25': float(q25),
            'Q75': float(q75),
            'IQR': float(iqr),
            '下界': float(lower_bound),
            '上界': float(upper_bound),
            '异常值索引': outlier_indices
        }
    
    return outliers_result


def check_time_continuity(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    expected_freq: str = '15min'
) -> Dict[str, Any]:
    """检查时间连续性（15分钟间隔）。

    Args:
        df: 输入的DataFrame。
        datetime_col: 时间列名，默认为'datetime'。
        expected_freq: 期望的时间间隔，默认为'15T'（15分钟）。

    Returns:
        时间连续性检查结果字典。
    """
    if datetime_col not in df.columns:
        return {
            '是否连续': False,
            '错误': f"DataFrame中不存在 {datetime_col} 列",
            '间隔数量': 0,
            '间隔列表': []
        }
    
    datetime_series = pd.to_datetime(df[datetime_col], errors='coerce')
    valid_datetime = datetime_series.dropna().sort_values().reset_index(drop=True)
    
    if len(valid_datetime) < 2:
        return {
            '是否连续': True,
            '时间范围': (valid_datetime.min(), valid_datetime.max()) if len(valid_datetime) > 0 else None,
            '数据点数': len(valid_datetime),
            '间隔数量': 0,
            '间隔列表': []
        }
    
    is_continuous, gaps = check_time_continuous(valid_datetime, expected_freq)
    
    expected_delta = pd.Timedelta(expected_freq)
    time_range = (valid_datetime.min(), valid_datetime.max())
    total_duration = time_range[1] - time_range[0]
    expected_points = int(total_duration / expected_delta) + 1
    
    return {
        '是否连续': is_continuous,
        '时间范围': time_range,
        '数据点数': len(valid_datetime),
        '期望点数': expected_points,
        '缺失点数': expected_points - len(valid_datetime),
        '间隔数量': len(gaps),
        '间隔列表': gaps
    }


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> Dict[str, Any]:
    """检测重复数据。

    Args:
        df: 输入的DataFrame。
        subset: 用于检测重复的列名列表，为None时使用所有列。
        keep: 保留策略，'first'保留第一个，'last'保留最后一个，False标记所有重复。

    Returns:
        重复数据检测结果字典。
    """
    duplicate_mask = df.duplicated(subset=subset, keep=keep)
    duplicate_count = int(duplicate_mask.sum())
    
    duplicate_indices = df.index[duplicate_mask].tolist()
    
    duplicate_rows = df[duplicate_mask].copy() if duplicate_count > 0 else pd.DataFrame()
    
    return {
        '重复数量': duplicate_count,
        '重复比例': duplicate_count / len(df) * 100 if len(df) > 0 else 0,
        '重复索引': duplicate_indices,
        '重复数据': duplicate_rows
    }


def calculate_quality_score(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    expected_freq: str = '15min',
    weights: Optional[Dict[str, float]] = None
) -> float:
    """计算综合数据质量评分（0-100分）。

    Args:
        df: 输入的DataFrame。
        datetime_col: 时间列名。
        expected_freq: 期望的时间间隔。
        weights: 各指标权重字典，默认为等权重。

    Returns:
        综合质量评分，范围0-100。
    """
    if weights is None:
        weights = {
            'missing': 0.3,
            'outliers': 0.25,
            'continuity': 0.25,
            'duplicates': 0.2
        }
    
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    scores: Dict[str, float] = {}
    
    missing_stats = calculate_missing_rate(df)
    avg_missing_rate = missing_stats['缺失率'].mean()
    scores['missing'] = max(0.0, 100.0 - avg_missing_rate * 2)
    
    outliers = detect_outliers(df)
    if outliers:
        avg_outlier_rate = np.mean([v['异常值比例'] for v in outliers.values()])
        scores['outliers'] = max(0.0, 100.0 - avg_outlier_rate * 3)
    else:
        scores['outliers'] = 100.0
    
    continuity = check_time_continuity(df, datetime_col, expected_freq)
    if continuity.get('是否连续', False):
        scores['continuity'] = 100.0
    else:
        expected_points = continuity.get('期望点数', 1)
        actual_points = continuity.get('数据点数', 0)
        continuity_score = (actual_points / expected_points * 100) if expected_points > 0 else 0
        gap_penalty = min(continuity.get('间隔数量', 0) * 5, 50)
        scores['continuity'] = max(0.0, continuity_score - gap_penalty)
    
    duplicates = detect_duplicates(df)
    scores['duplicates'] = max(0.0, 100.0 - duplicates['重复比例'] * 5)
    
    total_score = sum(scores[k] * normalized_weights[k] for k in normalized_weights)
    
    return round(total_score, 2)


def generate_quality_report(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    expected_freq: str = '15min',
    outlier_k: float = 1.5
) -> pd.DataFrame:
    """生成详细的质量检查报告（DataFrame格式）。

    Args:
        df: 输入的DataFrame。
        datetime_col: 时间列名。
        expected_freq: 期望的时间间隔。
        outlier_k: IQR异常值检测的k值。

    Returns:
        详细的质量检查报告DataFrame。
    """
    report_data: List[Dict[str, Any]] = []
    
    missing_stats = calculate_missing_rate(df)
    for _, row in missing_stats.iterrows():
        report_data.append({
            '检查项': '缺失值检查',
            '字段': row['字段名'],
            '指标': '缺失数量',
            '数值': row['缺失数量'],
            '描述': f"字段 {row['字段名']} 缺失 {row['缺失数量']} 条"
        })
        report_data.append({
            '检查项': '缺失值检查',
            '字段': row['字段名'],
            '指标': '缺失率(%)',
            '数值': round(row['缺失率'], 2),
            '描述': f"字段 {row['字段名']} 缺失率 {row['缺失率']:.2f}%"
        })
    
    outliers = detect_outliers(df, k=outlier_k)
    for col, info in outliers.items():
        report_data.append({
            '检查项': '异常值检查',
            '字段': col,
            '指标': '异常值数量',
            '数值': info['异常值数量'],
            '描述': f"字段 {col} 检测到 {info['异常值数量']} 个异常值"
        })
        report_data.append({
            '检查项': '异常值检查',
            '字段': col,
            '指标': '异常值比例(%)',
            '数值': round(info['异常值比例'], 2),
            '描述': f"字段 {col} 异常值比例 {info['异常值比例']:.2f}%"
        })
        report_data.append({
            '检查项': '异常值检查',
            '字段': col,
            '指标': '下界',
            '数值': round(info['下界'], 2),
            '描述': f"字段 {col} 正常值下界 {info['下界']:.2f}"
        })
        report_data.append({
            '检查项': '异常值检查',
            '字段': col,
            '指标': '上界',
            '数值': round(info['上界'], 2),
            '描述': f"字段 {col} 正常值上界 {info['上界']:.2f}"
        })
    
    continuity = check_time_continuity(df, datetime_col, expected_freq)
    report_data.append({
        '检查项': '时间连续性检查',
        '字段': datetime_col,
        '指标': '是否连续',
        '数值': 1 if continuity.get('是否连续', False) else 0,
        '描述': '时间序列' + ('连续' if continuity.get('是否连续', False) else '不连续')
    })
    report_data.append({
        '检查项': '时间连续性检查',
        '字段': datetime_col,
        '指标': '数据点数',
        '数值': continuity.get('数据点数', 0),
        '描述': f"有效数据点 {continuity.get('数据点数', 0)} 个"
    })
    report_data.append({
        '检查项': '时间连续性检查',
        '字段': datetime_col,
        '指标': '间隔数量',
        '数值': continuity.get('间隔数量', 0),
        '描述': f"检测到 {continuity.get('间隔数量', 0)} 个时间间隔"
    })
    if continuity.get('时间范围'):
        time_range = continuity['时间范围']
        report_data.append({
            '检查项': '时间连续性检查',
            '字段': datetime_col,
            '指标': '开始时间',
            '数值': None,
            '描述': f"时间范围开始于 {time_range[0]}"
        })
        report_data.append({
            '检查项': '时间连续性检查',
            '字段': datetime_col,
            '指标': '结束时间',
            '数值': None,
            '描述': f"时间范围结束于 {time_range[1]}"
        })
    
    duplicates = detect_duplicates(df)
    report_data.append({
        '检查项': '重复数据检查',
        '字段': '全部',
        '指标': '重复数量',
        '数值': duplicates['重复数量'],
        '描述': f"检测到 {duplicates['重复数量']} 条重复数据"
    })
    report_data.append({
        '检查项': '重复数据检查',
        '字段': '全部',
        '指标': '重复比例(%)',
        '数值': round(duplicates['重复比例'], 2),
        '描述': f"重复数据比例 {duplicates['重复比例']:.2f}%"
    })
    
    quality_score = calculate_quality_score(df, datetime_col, expected_freq)
    report_data.append({
        '检查项': '综合评分',
        '字段': '全部',
        '指标': '质量评分',
        '数值': quality_score,
        '描述': f"综合数据质量评分为 {quality_score} 分"
    })
    
    report_df = pd.DataFrame(report_data)
    return report_df


def fill_missing_values(
    df: pd.DataFrame,
    strategy: str = 'interpolate',
    columns: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    method: str = 'time'
) -> pd.DataFrame:
    """填充缺失值。

    Args:
        df: 输入的DataFrame。
        strategy: 填充策略，可选 'interpolate'、'mean'、'median'、'ffill'、'bfill'、'value'。
        columns: 需要填充的列名列表，为None时填充所有列。
        fill_value: 当 strategy='value' 时使用的填充值。
        method: 插值方法，当 strategy='interpolate' 时使用。

    Returns:
        填充缺失值后的DataFrame副本。
    """
    df = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    other_cols = [col for col in columns if col in df.columns and col not in numeric_cols]
    
    if strategy == 'value':
        if fill_value is None:
            raise ValueError("当 strategy='value' 时必须提供 fill_value")
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
        df[other_cols] = df[other_cols].fillna(fill_value)
    
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    elif strategy == 'ffill':
        df[columns] = df[columns].ffill()
    
    elif strategy == 'bfill':
        df[columns] = df[columns].bfill()
    
    elif strategy == 'interpolate':
        if 'datetime' in df.columns and method == 'time':
            df_temp = df.set_index('datetime')
            df_temp[numeric_cols] = df_temp[numeric_cols].interpolate(method='time')
            df[numeric_cols] = df_temp[numeric_cols].values
        else:
            df[numeric_cols] = df[numeric_cols].interpolate(method=method)
        df[other_cols] = df[other_cols].ffill().bfill()
    
    else:
        raise ValueError(f"不支持的填充策略: {strategy}")
    
    return df


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    k: float = 1.5,
    method: str = 'clip',
    fill_value: Optional[Any] = None
) -> pd.DataFrame:
    """处理异常值。

    Args:
        df: 输入的DataFrame。
        columns: 需要处理的列名列表，为None时处理所有数值列。
        k: IQR倍数，默认为1.5。
        method: 处理方法，可选 'clip'（截断）、'remove'（删除）、'nan'（设为NaN）、'value'（替换为指定值）。
        fill_value: 当 method='value' 时使用的替换值。

    Returns:
        处理异常值后的DataFrame副本。
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = detect_outliers(df, columns, k)
    
    for col, info in outliers.items():
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        lower_bound = info['下界']
        upper_bound = info['上界']
        
        if method == 'clip':
            df[col] = series.clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'remove':
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            df.loc[outlier_mask, col] = np.nan
        
        elif method == 'nan':
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            df.loc[outlier_mask, col] = np.nan
        
        elif method == 'value':
            if fill_value is None:
                raise ValueError("当 method='value' 时必须提供 fill_value")
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            df.loc[outlier_mask, col] = fill_value
        
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")
    
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """删除重复数据。

    Args:
        df: 输入的DataFrame。
        subset: 用于检测重复的列名列表，为None时使用所有列。
        keep: 保留策略，'first'保留第一个，'last'保留最后一个。

    Returns:
        删除重复数据后的DataFrame副本。
    """
    df = df.copy()
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    return df


def clean_data(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    remove_dupes: bool = True,
    handle_outliers_flag: bool = True,
    fill_missing_flag: bool = True,
    outlier_k: float = 1.5,
    outlier_method: str = 'clip',
    missing_strategy: str = 'interpolate',
    **kwargs: Any
) -> pd.DataFrame:
    """综合数据清洗功能。

    Args:
        df: 输入的DataFrame。
        datetime_col: 时间列名。
        remove_dupes: 是否删除重复数据。
        handle_outliers_flag: 是否处理异常值。
        fill_missing_flag: 是否填充缺失值。
        outlier_k: IQR异常值检测的k值。
        outlier_method: 异常值处理方法。
        missing_strategy: 缺失值填充策略。
        **kwargs: 传递给各清洗函数的额外参数。

    Returns:
        清洗后的DataFrame副本。
    """
    df = df.copy()
    
    if remove_dupes:
        df = remove_duplicates(df, subset=kwargs.get('duplicate_subset'), keep=kwargs.get('keep', 'first'))
    
    if handle_outliers_flag:
        df = handle_outliers(
            df,
            columns=kwargs.get('outlier_columns'),
            k=outlier_k,
            method=outlier_method,
            fill_value=kwargs.get('outlier_fill_value')
        )
    
    if fill_missing_flag:
        df = fill_missing_values(
            df,
            strategy=missing_strategy,
            columns=kwargs.get('fill_columns'),
            fill_value=kwargs.get('fill_value'),
            method=kwargs.get('interpolate_method', 'time')
        )
    
    return df
