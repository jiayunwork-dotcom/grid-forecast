import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_load_profile(start_date: str, days: int = 30) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=days * 96, freq='15min')
    n = len(dates)
    
    base_load = 80.0
    hourly_pattern = np.array([
        0.75, 0.72, 0.70, 0.68, 0.67, 0.68, 0.72, 0.78,
        0.88, 0.95, 1.00, 1.02, 1.00, 0.98, 0.96, 0.94,
        0.92, 0.93, 0.95, 0.97, 0.98, 0.96, 0.90, 0.82
    ])
    hourly_pattern = np.repeat(hourly_pattern, 4)
    daily_pattern = np.tile(hourly_pattern, days)
    
    weekday_factor = np.where(dates.dayofweek < 5, 1.0, 0.9)
    
    monthly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (dates.month - 6) / 12)
    
    temperature = generate_temperature(dates)
    temp_effect = 1.0 + 0.008 * (temperature - 20) - 0.0002 * (temperature - 20) ** 2
    
    noise = np.random.normal(0, 0.02, n)
    
    active_power = base_load * daily_pattern * weekday_factor * monthly_factor * temp_effect * (1 + noise)
    active_power = np.maximum(active_power, base_load * 0.5)
    
    reactive_power = active_power * (0.45 + 0.05 * np.random.normal(0, 0.1, n))
    
    voltage = 110.0 + np.random.normal(0, 1.5, n)
    
    humidity = generate_humidity(dates)
    solar = generate_solar_irradiance(dates)
    wind_speed = generate_wind_speed(dates)
    
    df = pd.DataFrame({
        'datetime': dates,
        'active_power_MW': active_power.round(2),
        'reactive_power_MVar': reactive_power.round(2),
        'voltage_kV': voltage.round(2),
        'temperature_C': temperature.round(1),
        'humidity_pct': humidity.round(1),
        'solar_irradiance_Wm2': solar.round(1),
        'wind_speed_ms': wind_speed.round(2)
    })
    
    return df


def generate_temperature(dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(dates)
    month = dates.month
    hour = dates.hour
    
    base_temp = 15.0 + 15.0 * np.sin(2 * np.pi * (month - 7) / 12)
    
    daily_cycle = -6.0 * np.cos(2 * np.pi * (hour - 14) / 24)
    
    weather_noise = np.random.normal(0, 2.0, n)
    
    temp = base_temp + daily_cycle + weather_noise
    return np.clip(temp, -10, 42)


def generate_humidity(dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(dates)
    hour = dates.hour
    
    base_humidity = 60.0 + 15.0 * np.sin(2 * np.pi * (dates.month - 7) / 12)
    
    daily_cycle = 10.0 * np.cos(2 * np.pi * (hour - 6) / 24)
    
    noise = np.random.normal(0, 8, n)
    
    humidity = base_humidity + daily_cycle + noise
    return np.clip(humidity, 10, 98)


def generate_solar_irradiance(dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(dates)
    hour = dates.hour + dates.minute / 60
    
    day_of_year = dates.dayofyear
    declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    
    latitude = 35.0
    hour_angle = 15.0 * (hour - 12)
    
    cos_zenith = (np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
                  np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
                  np.cos(np.radians(hour_angle)))
    
    solar = 1000.0 * np.maximum(cos_zenith, 0) ** 1.2
    
    cloud_factor = np.random.uniform(0.3, 1.0, n)
    cloud_factor = np.repeat(cloud_factor[::96], 96)[:n]
    
    solar = solar * cloud_factor
    return solar


def generate_wind_speed(dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(dates)
    hour = dates.hour
    
    base_wind = 4.0 + 2.0 * np.sin(2 * np.pi * dates.month / 12)
    
    daily_cycle = 1.5 * np.sin(2 * np.pi * (hour - 14) / 24)
    
    gusts = np.maximum(0, np.random.normal(0, 1.0, n))
    
    wind = base_wind + daily_cycle + gusts
    return np.clip(wind, 0, 25)


def generate_calendar_data(start_date: str, days: int = 30) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=days, freq='D')
    
    day_types = []
    is_holidays = []
    special_events = []
    
    holidays_2025 = [
        '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31',
        '2025-02-01', '2025-02-02', '2025-04-04', '2025-04-05', '2025-04-06',
        '2025-05-01', '2025-05-31', '2025-06-01', '2025-06-02', '2025-10-01',
        '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06',
        '2025-10-07'
    ]
    
    special_days = {
        '2025-02-14': '情人节',
        '2025-06-18': '电商促销',
        '2025-11-11': '双十一购物节',
        '2025-12-25': '圣诞节'
    }
    
    for dt in dates:
        date_str = dt.strftime('%Y-%m-%d')
        dow = dt.dayofweek
        
        if date_str in holidays_2025:
            day_types.append('节假日')
            is_holidays.append(True)
        elif dow >= 5:
            day_types.append('周末')
            is_holidays.append(False)
        else:
            day_types.append('工作日')
            is_holidays.append(False)
        
        special_events.append(special_days.get(date_str, ''))
    
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'day_type': day_types,
        'is_holiday': is_holidays,
        'special_event': special_events
    })
    
    return df


def generate_forecast_temperatures(dates: pd.DatetimeIndex, hours_ahead: int = 24) -> pd.DataFrame:
    actual_temps = generate_temperature(dates)
    forecast_error = np.random.normal(0, 1.5, len(dates))
    forecast_temps = actual_temps + forecast_error
    
    df = pd.DataFrame({
        'datetime': dates,
        'temperature_forecast_C': forecast_temps.round(1)
    })
    
    return df


def add_data_quality_issues(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = df.copy()
    n = len(df)
    
    n_missing = int(n * 0.01)
    missing_indices = np.random.choice(n, n_missing, replace=False)
    df.loc[missing_indices, 'active_power_MW'] = np.nan
    
    n_outliers = int(n * 0.005)
    outlier_indices = np.random.choice(n, n_outliers, replace=False)
    df.loc[outlier_indices, 'active_power_MW'] *= 1.5
    
    n_duplicates = 10
    dup_indices = np.random.choice(n - 1, n_duplicates, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    start_date = '2025-06-01'
    days = 60
    
    print("生成负荷气象数据...")
    load_df = generate_load_profile(start_date, days)
    load_df = add_data_quality_issues(load_df)
    load_path = os.path.join(data_dir, 'load_data.csv')
    load_df.to_csv(load_path, index=False, encoding='utf-8-sig')
    print(f"  已保存: {load_path}")
    print(f"  数据范围: {load_df['datetime'].min()} 至 {load_df['datetime'].max()}")
    print(f"  数据点数: {len(load_df)}")
    
    print("\n生成日历信息表...")
    calendar_df = generate_calendar_data(start_date, days)
    calendar_path = os.path.join(data_dir, 'calendar_data.csv')
    calendar_df.to_csv(calendar_path, index=False, encoding='utf-8-sig')
    print(f"  已保存: {calendar_path}")
    print(f"  日期范围: {calendar_df['date'].min()} 至 {calendar_df['date'].max()}")
    print(f"  工作日: {(calendar_df['day_type'] == '工作日').sum()} 天")
    print(f"  周末: {(calendar_df['day_type'] == '周末').sum()} 天")
    print(f"  节假日: {(calendar_df['day_type'] == '节假日').sum()} 天")
    
    print("\n生成气温预报数据...")
    dates = pd.DatetimeIndex(load_df['datetime'])
    forecast_df = generate_forecast_temperatures(dates)
    forecast_path = os.path.join(data_dir, 'temperature_forecast.csv')
    forecast_df.to_csv(forecast_path, index=False, encoding='utf-8-sig')
    print(f"  已保存: {forecast_path}")
    
    print("\n=== 示例数据生成完成 ===")
    print(f"数据目录: {data_dir}")


if __name__ == '__main__':
    main()
