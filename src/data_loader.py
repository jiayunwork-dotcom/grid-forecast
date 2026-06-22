import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, IO, Tuple
from datetime import datetime


STANDARD_COLUMN_MAPPING: Dict[str, List[str]] = {
    'datetime': ['datetime', '时间', '日期时间', 'timestamp', 'date_time', 'dt'],
    'load': ['load', '负荷', '用电负荷', 'power', 'power_load', 'active_power'],
    'temperature': ['temperature', '温度', '气温', 'temp', 't'],
    'humidity': ['humidity', '湿度', '相对湿度', 'rh', 'h'],
    'pressure': ['pressure', '气压', '大气压', 'pres', 'p'],
    'wind_speed': ['wind_speed', '风速', '风力', 'ws', 'wind'],
    'wind_direction': ['wind_direction', '风向', 'wd'],
    'precipitation': ['precipitation', '降水量', '降雨', 'precip', 'rain'],
    'sunshine': ['sunshine', '日照', '日照时数', 'sun'],
    'holiday': ['holiday', '节假日', '是否节假日', 'is_holiday'],
    'workday': ['workday', '工作日', '是否工作日', 'is_workday'],
    'date': ['date', '日期', 'dt_date'],
}

NUMERIC_FIELDS: List[str] = [
    'load', 'temperature', 'humidity', 'pressure', 'wind_speed',
    'wind_direction', 'precipitation', 'sunshine', 'holiday', 'workday'
]

DATETIME_FORMATS: List[str] = [
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y/%m/%d %H:%M',
    '%Y-%m-%d',
    '%Y/%m/%d',
    '%Y%m%d%H%M%S',
    '%Y%m%d%H%M',
]

REQUIRED_FIELDS_LOAD: List[str] = ['datetime', 'load']
REQUIRED_FIELDS_CALENDAR: List[str] = ['date']

VALID_RANGES: Dict[str, Tuple[float, float]] = {
    'temperature': (-40.0, 50.0),
    'humidity': (0.0, 100.0),
    'pressure': (800.0, 1100.0),
    'wind_speed': (0.0, 100.0),
    'wind_direction': (0.0, 360.0),
    'precipitation': (0.0, 1000.0),
    'sunshine': (0.0, 24.0),
    'holiday': (0.0, 1.0),
    'workday': (0.0, 1.0),
}


def get_standard_column_name(col_name: str, mapping: Optional[Dict[str, List[str]]] = None) -> str:
    """获取标准化的列名。

    Args:
        col_name: 原始列名。
        mapping: 列名映射字典，默认为 STANDARD_COLUMN_MAPPING。

    Returns:
        标准化后的列名，如果没有匹配则返回原始列名。
    """
    if mapping is None:
        mapping = STANDARD_COLUMN_MAPPING
    
    col_lower = col_name.strip().lower()
    for standard_name, aliases in mapping.items():
        if col_lower in [alias.lower() for alias in aliases]:
            return standard_name
    return col_name


def standardize_column_names(df: pd.DataFrame, mapping: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """标准化DataFrame的列名。

    Args:
        df: 输入的DataFrame。
        mapping: 列名映射字典，默认为 STANDARD_COLUMN_MAPPING。

    Returns:
        列名标准化后的DataFrame副本。
    """
    df = df.copy()
    new_columns = {}
    for col in df.columns:
        new_columns[col] = get_standard_column_name(col, mapping)
    df = df.rename(columns=new_columns)
    return df


def parse_datetime_series(
    datetime_series: pd.Series,
    formats: Optional[List[str]] = None
) -> pd.Series:
    """解析时间序列，支持多种时间格式。

    Args:
        datetime_series: 包含时间字符串的Series。
        formats: 时间格式列表，默认为 DATETIME_FORMATS。

    Returns:
        解析后的datetime64类型Series，无法解析的值为NaT。
    """
    if formats is None:
        formats = DATETIME_FORMATS
    
    datetime_series = datetime_series.astype(str).str.strip()
    result = pd.Series([pd.NaT] * len(datetime_series), index=datetime_series.index)
    
    for fmt in formats:
        mask = result.isna()
        if not mask.any():
            break
        try:
            parsed = pd.to_datetime(datetime_series[mask], format=fmt, errors='coerce')
            result[mask] = parsed
        except (ValueError, TypeError):
            continue
    
    remaining_mask = result.isna()
    if remaining_mask.any():
        result[remaining_mask] = pd.to_datetime(datetime_series[remaining_mask], errors='coerce')
    
    return result


def auto_detect_field_types(df: pd.DataFrame) -> Dict[str, str]:
    """自动识别DataFrame的字段类型。

    Args:
        df: 输入的DataFrame。

    Returns:
        字段类型字典，键为列名，值为'datetime'、'numeric'、'date'或'categorical'。
    """
    field_types: Dict[str, str] = {}
    
    for col in df.columns:
        if col == 'datetime' or 'time' in col.lower() or '日期' in col:
            field_types[col] = 'datetime'
        elif col in NUMERIC_FIELDS or pd.api.types.is_numeric_dtype(df[col]):
            field_types[col] = 'numeric'
        elif col == 'date':
            field_types[col] = 'date'
        else:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                field_types[col] = 'datetime'
            elif pd.api.types.is_numeric_dtype(df[col]):
                field_types[col] = 'numeric'
            else:
                field_types[col] = 'categorical'
    
    return field_types


def convert_field_types(
    df: pd.DataFrame,
    field_types: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """转换DataFrame的字段类型。

    Args:
        df: 输入的DataFrame。
        field_types: 字段类型字典，为None时自动检测。

    Returns:
        类型转换后的DataFrame副本。
    """
    df = df.copy()
    
    if field_types is None:
        field_types = auto_detect_field_types(df)
    
    for col, field_type in field_types.items():
        if col not in df.columns:
            continue
        
        if field_type == 'datetime':
            df[col] = parse_datetime_series(df[col])
        elif field_type == 'date':
            df[col] = parse_datetime_series(df[col]).dt.date
        elif field_type == 'numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def check_required_fields(
    df: pd.DataFrame,
    required_fields: List[str]
) -> Tuple[bool, List[str]]:
    """检查DataFrame是否包含必填字段。

    Args:
        df: 输入的DataFrame。
        required_fields: 必填字段列表。

    Returns:
        元组(是否包含所有必填字段, 缺失的字段列表)。
    """
    missing_fields = [field for field in required_fields if field not in df.columns]
    return (len(missing_fields) == 0, missing_fields)


def check_value_ranges(
    df: pd.DataFrame,
    valid_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Dict[str, Any]]:
    """检查数值字段是否在有效范围内。

    Args:
        df: 输入的DataFrame。
        valid_ranges: 有效范围字典，键为列名，值为(最小值, 最大值)元组。

    Returns:
        范围违规信息字典，键为列名，值为包含count、valid_range和indices的字典。
    """
    if valid_ranges is None:
        valid_ranges = VALID_RANGES
    
    range_violations: Dict[str, Dict[str, Any]] = {}
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors='coerce')
        violations = (series < min_val) | (series > max_val)
        violation_count = violations.sum()
        
        if violation_count > 0:
            violation_indices = df.index[violations].tolist()
            range_violations[col] = {
                'count': int(violation_count),
                'valid_range': (min_val, max_val),
                'indices': violation_indices
            }
    
    return range_violations


def validate_data(
    df: pd.DataFrame,
    data_type: str = 'load',
    valid_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """综合验证数据质量。

    Args:
        df: 输入的DataFrame。
        data_type: 数据类型，'load'表示负荷气象数据，'calendar'表示日历信息。
        valid_ranges: 有效范围字典。

    Returns:
        验证结果字典，包含is_valid、errors和warnings。
    """
    if data_type == 'load':
        required_fields = REQUIRED_FIELDS_LOAD
    elif data_type == 'calendar':
        required_fields = REQUIRED_FIELDS_CALENDAR
    else:
        required_fields = []
    
    validation_result: Dict[str, Any] = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    has_required, missing_fields = check_required_fields(df, required_fields)
    if not has_required:
        validation_result['is_valid'] = False
        validation_result['errors'].append(
            f"缺少必填字段: {', '.join(missing_fields)}"
        )
    
    range_violations = check_value_ranges(df, valid_ranges)
    for col, info in range_violations.items():
        validation_result['warnings'].append(
            f"字段 {col} 有 {info['count']} 个值超出有效范围 {info['valid_range']}"
        )
    
    if 'datetime' in df.columns:
        null_datetime = df['datetime'].isna().sum()
        if null_datetime > 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append(
                f"datetime 字段有 {null_datetime} 个无效时间戳"
            )
    
    if 'load' in df.columns:
        load_series = pd.to_numeric(df['load'], errors='coerce')
        negative_load = (load_series < 0).sum()
        if negative_load > 0:
            validation_result['warnings'].append(
                f"负荷字段有 {negative_load} 个负值"
            )
        null_load = load_series.isna().sum()
        if null_load > 0:
            validation_result['warnings'].append(
                f"负荷字段有 {null_load} 个缺失值"
            )
    
    return validation_result


def sort_by_datetime(
    df: pd.DataFrame,
    datetime_col: str = 'datetime',
    ascending: bool = True
) -> pd.DataFrame:
    """按时间列排序DataFrame。

    Args:
        df: 输入的DataFrame。
        datetime_col: 时间列名，默认为'datetime'。
        ascending: 是否升序排列，默认为True。

    Returns:
        排序后的DataFrame副本。

    Raises:
        ValueError: 当DataFrame中不存在指定的时间列时抛出。
    """
    df = df.copy()
    if datetime_col not in df.columns:
        raise ValueError(f"DataFrame 中不存在 {datetime_col} 列")
    
    df = df.sort_values(by=datetime_col, ascending=ascending).reset_index(drop=True)
    return df


def read_csv_file(
    filepath: str,
    encoding: str = 'utf-8',
    standardize_names: bool = True,
    convert_types: bool = True,
    sort_by_time: bool = True,
    datetime_col: Optional[str] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """读取CSV文件并进行标准化处理。

    Args:
        filepath: CSV文件路径。
        encoding: 文件编码，默认为'utf-8'。
        standardize_names: 是否标准化列名，默认为True。
        convert_types: 是否转换字段类型，默认为True。
        sort_by_time: 是否按时间排序，默认为True。
        datetime_col: 时间列名，为None时自动检测。
        **kwargs: 传递给pd.read_csv的其他参数。

    Returns:
        处理后的DataFrame。
    """
    df = pd.read_csv(filepath, encoding=encoding, **kwargs)
    
    if standardize_names:
        df = standardize_column_names(df)
    
    if convert_types:
        df = convert_field_types(df)
    
    if sort_by_time:
        if datetime_col is None:
            datetime_col = 'datetime' if 'datetime' in df.columns else None
        if datetime_col and datetime_col in df.columns:
            df = sort_by_datetime(df, datetime_col)
    
    return df


def read_csv_from_fileobj(
    fileobj: Union[IO[bytes], IO[str]],
    encoding: str = 'utf-8',
    standardize_names: bool = True,
    convert_types: bool = True,
    sort_by_time: bool = True,
    datetime_col: Optional[str] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """从文件对象读取CSV数据并进行标准化处理。

    Args:
        fileobj: 文件对象，可以是二进制或文本模式。
        encoding: 文件编码，默认为'utf-8'。
        standardize_names: 是否标准化列名，默认为True。
        convert_types: 是否转换字段类型，默认为True。
        sort_by_time: 是否按时间排序，默认为True。
        datetime_col: 时间列名，为None时自动检测。
        **kwargs: 传递给pd.read_csv的其他参数。

    Returns:
        处理后的DataFrame。
    """
    df = pd.read_csv(fileobj, encoding=encoding, **kwargs)
    
    if standardize_names:
        df = standardize_column_names(df)
    
    if convert_types:
        df = convert_field_types(df)
    
    if sort_by_time:
        if datetime_col is None:
            datetime_col = 'datetime' if 'datetime' in df.columns else None
        if datetime_col and datetime_col in df.columns:
            df = sort_by_datetime(df, datetime_col)
    
    return df


def load_load_data(
    filepath: str,
    encoding: str = 'utf-8',
    validate: bool = True,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """加载15分钟粒度负荷气象数据。

    Args:
        filepath: CSV文件路径。
        encoding: 文件编码，默认为'utf-8'。
        validate: 是否进行数据验证，默认为True。
        **kwargs: 传递给read_csv_file的其他参数。

    Returns:
        元组(处理后的DataFrame, 验证结果字典)。
    """
    df = read_csv_file(filepath, encoding=encoding, **kwargs)
    
    validation_result = {'is_valid': True, 'errors': [], 'warnings': []}
    if validate:
        validation_result = validate_data(df, data_type='load')
    
    return df, validation_result


def load_calendar_data(
    filepath: str,
    encoding: str = 'utf-8',
    validate: bool = True,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """加载日历信息表。

    Args:
        filepath: CSV文件路径。
        encoding: 文件编码，默认为'utf-8'。
        validate: 是否进行数据验证，默认为True。
        **kwargs: 传递给read_csv_file的其他参数。

    Returns:
        元组(处理后的DataFrame, 验证结果字典)。
    """
    df = read_csv_file(filepath, encoding=encoding, **kwargs)
    
    validation_result = {'is_valid': True, 'errors': [], 'warnings': []}
    if validate:
        validation_result = validate_data(df, data_type='calendar')
    
    return df, validation_result
