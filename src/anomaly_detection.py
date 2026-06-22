import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta


class AnomalyDetector:
    """负荷异常检测与事件关联分析类"""

    def __init__(
        self,
        target_col: str = 'active_power_MW',
        datetime_col: str = 'datetime',
        temp_col: str = 'temperature_C',
        solar_col: str = 'solar_irradiance_Wm2',
        baseline_days: int = 14,
        mad_threshold: float = 3.0,
        max_iterations: int = 3,
        max_gap_minutes: int = 45,
        temp_window: str = '1h',
        temp_threshold: float = 3.0,
        solar_threshold: float = 200.0,
        correlation_window_hours: int = 2
    ) -> None:
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.temp_col = temp_col
        self.solar_col = solar_col
        self.baseline_days = baseline_days
        self.mad_threshold = mad_threshold
        self.max_iterations = max_iterations
        self.max_gap_minutes = max_gap_minutes
        self.temp_window = temp_window
        self.temp_threshold = temp_threshold
        self.solar_threshold = solar_threshold
        self.correlation_window_hours = correlation_window_hours

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.sort_values(self.datetime_col).reset_index(drop=True)
        return df

    def _compute_baseline_for_point(
        self,
        series: pd.Series,
        times: pd.Series,
        idx: int
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        current_time = times.iloc[idx]
        target_time = current_time.time()

        start_lookback = current_time - timedelta(days=self.baseline_days)
        end_lookback = current_time - timedelta(minutes=1)

        mask = (
            (times >= start_lookback) &
            (times <= end_lookback) &
            (times.dt.time == target_time)
        )

        history_values = series[mask].values
        history_values = history_values[~np.isnan(history_values)]
        baseline_sample_count = len(history_values)

        if baseline_sample_count < 3:
            return None, None, None

        for _ in range(self.max_iterations):
            median_val = np.median(history_values)
            mad = np.median(np.abs(history_values - median_val))
            std_est = 1.4826 * mad if mad > 0 else np.std(history_values)
            if std_est == 0:
                break
            threshold = self.mad_threshold * std_est
            deviations = np.abs(history_values - median_val)
            keep_mask = deviations <= threshold
            if keep_mask.sum() == len(history_values):
                break
            if keep_mask.sum() < 3:
                break
            history_values = history_values[keep_mask]

        median_val = np.median(history_values)
        mad = np.median(np.abs(history_values - median_val))
        std_est = 1.4826 * mad if mad > 0 else (np.std(history_values) if len(history_values) > 0 else 0)

        return float(median_val), float(std_est), baseline_sample_count

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_data(df)

        result_df = df.copy()
        result_df['baseline_median'] = np.nan
        result_df['baseline_std'] = np.nan
        result_df['baseline_sample_count'] = np.nan
        result_df['is_anomaly'] = False
        result_df['deviation_ratio'] = np.nan
        result_df['deviation_direction'] = 0

        series = df[self.target_col].astype(float)
        times = df[self.datetime_col]

        for i in range(len(df)):
            median_val, std_val, sample_count = self._compute_baseline_for_point(series, times, i)
            if median_val is None or std_val is None or std_val == 0:
                continue

            actual_val = float(series.iloc[i])
            deviation = actual_val - median_val
            deviation_ratio = deviation / std_val

            result_df.at[i, 'baseline_median'] = median_val
            result_df.at[i, 'baseline_std'] = std_val
            result_df.at[i, 'baseline_sample_count'] = sample_count
            result_df.at[i, 'deviation_ratio'] = deviation_ratio

            if abs(deviation_ratio) > self.mad_threshold:
                result_df.at[i, 'is_anomaly'] = True
                result_df.at[i, 'deviation_direction'] = 1 if deviation_ratio > 0 else -1

        return result_df

    def _calculate_confidence_score(
        self,
        peak_deviation_ratio: float,
        duration_minutes: float,
        avg_baseline_samples: float,
        mad_threshold: float
    ) -> int:
        deviation_factor = min(abs(peak_deviation_ratio) / mad_threshold * 40, 40)
        duration_factor = min(duration_minutes / 120 * 30, 30)
        sample_factor = min(avg_baseline_samples / 14 * 30, 30)
        total_score = deviation_factor + duration_factor + sample_factor
        return int(round(total_score))

    def aggregate_anomaly_events(self, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        df = anomaly_df.copy()
        df = df.sort_values(self.datetime_col).reset_index(drop=True)

        anomaly_indices = df[df['is_anomaly']].index.tolist()
        if not anomaly_indices:
            return pd.DataFrame(columns=[
                'event_id', 'start_time', 'end_time', 'duration_minutes',
                'peak_deviation_ratio', 'deviation_direction', 'anomaly_count',
                'avg_baseline_samples', 'confidence_score'
            ])

        events = []
        current_event_start_idx = anomaly_indices[0]
        current_event_indices = [anomaly_indices[0]]

        for i in range(1, len(anomaly_indices)):
            prev_idx = anomaly_indices[i - 1]
            curr_idx = anomaly_indices[i]

            time_diff = (df.iloc[curr_idx][self.datetime_col] -
                         df.iloc[prev_idx][self.datetime_col]).total_seconds() / 60

            gap_points = curr_idx - prev_idx - 1
            if time_diff <= self.max_gap_minutes and gap_points <= 2:
                current_event_indices.append(curr_idx)
            else:
                events.append(current_event_indices)
                current_event_start_idx = curr_idx
                current_event_indices = [curr_idx]

        events.append(current_event_indices)

        event_list = []
        for event_id, event_indices in enumerate(events, start=1):
            event_data = df.iloc[event_indices]
            start_time = event_data[self.datetime_col].iloc[0]
            end_time = event_data[self.datetime_col].iloc[-1]
            duration = (end_time - start_time).total_seconds() / 60 + 15

            peak_idx = event_data['deviation_ratio'].abs().idxmax()
            peak_ratio = event_data.loc[peak_idx, 'deviation_ratio']
            direction = 1 if peak_ratio > 0 else -1

            avg_baseline_samples = event_data['baseline_sample_count'].mean()
            if pd.isna(avg_baseline_samples):
                avg_baseline_samples = 3.0

            confidence_score = self._calculate_confidence_score(
                peak_ratio, duration, avg_baseline_samples, self.mad_threshold
            )

            event_list.append({
                'event_id': event_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': int(duration),
                'peak_deviation_ratio': float(peak_ratio),
                'deviation_direction': '正异常(负荷突增)' if direction > 0 else '负异常(负荷骤降)',
                'anomaly_count': len(event_indices),
                'avg_baseline_samples': float(avg_baseline_samples),
                'confidence_score': confidence_score
            })

        return pd.DataFrame(event_list)

    def detect_weather_mutations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_data(df)

        result_df = df.copy()
        result_df['temp_mutation'] = False
        result_df['solar_mutation'] = False
        result_df['temp_diff'] = np.nan
        result_df['solar_diff'] = np.nan

        if self.temp_col in df.columns:
            times = df[self.datetime_col].values
            temp_vals = df[self.temp_col].astype(float).values
            temp_diff = np.full(len(temp_vals), np.nan)
            one_hour = np.timedelta64(1, 'h')

            for i in range(len(times)):
                target_time = times[i] - one_hour
                j = i - 1
                found = False
                while j >= 0:
                    time_diff = times[i] - times[j]
                    if time_diff >= one_hour - np.timedelta64(15, 'm') and time_diff <= one_hour + np.timedelta64(15, 'm'):
                        if not np.isnan(temp_vals[i]) and not np.isnan(temp_vals[j]):
                            temp_diff[i] = temp_vals[i] - temp_vals[j]
                        found = True
                        break
                    if time_diff > one_hour + np.timedelta64(15, 'm'):
                        break
                    j -= 1
                if not found and i >= 4:
                    if not np.isnan(temp_vals[i]) and not np.isnan(temp_vals[i - 4]):
                        temp_diff[i] = temp_vals[i] - temp_vals[i - 4]

            result_df['temp_diff'] = temp_diff
            result_df['temp_mutation'] = np.abs(result_df['temp_diff']) > self.temp_threshold

        if self.solar_col in df.columns:
            solar_vals = df[self.solar_col].astype(float).values
            solar_diff = np.full(len(solar_vals), np.nan)
            solar_diff[1:] = np.diff(solar_vals)
            result_df['solar_diff'] = solar_diff
            result_df['solar_mutation'] = np.abs(result_df['solar_diff']) > self.solar_threshold

        return result_df

    def aggregate_weather_events(self, weather_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        result = {}

        for event_type, col, diff_col in [
            ('temperature', 'temp_mutation', 'temp_diff'),
            ('solar', 'solar_mutation', 'solar_diff')
        ]:
            if col not in weather_df.columns:
                result[event_type] = pd.DataFrame()
                continue

            df = weather_df.sort_values(self.datetime_col).reset_index(drop=True)
            mutation_indices = df[df[col]].index.tolist()

            if not mutation_indices:
                result[event_type] = pd.DataFrame(columns=[
                    'event_id', 'start_time', 'end_time', 'duration_minutes',
                    'mutation_magnitude', 'mutation_points'
                ])
                continue

            events = []
            current_event = [mutation_indices[0]]

            for i in range(1, len(mutation_indices)):
                prev_idx = mutation_indices[i - 1]
                curr_idx = mutation_indices[i]
                gap = curr_idx - prev_idx
                if gap <= 2:
                    current_event.append(curr_idx)
                else:
                    events.append(current_event)
                    current_event = [curr_idx]
            events.append(current_event)

            event_list = []
            for eid, e_indices in enumerate(events, start=1):
                e_data = df.iloc[e_indices]
                start_t = e_data[self.datetime_col].iloc[0]
                end_t = e_data[self.datetime_col].iloc[-1]
                dur = (end_t - start_t).total_seconds() / 60 + 15

                if diff_col in e_data.columns:
                    max_mutation = e_data[diff_col].abs().max()
                else:
                    max_mutation = np.nan

                event_list.append({
                    'event_id': eid,
                    'start_time': start_t,
                    'end_time': end_t,
                    'duration_minutes': int(dur),
                    'mutation_magnitude': float(max_mutation) if not pd.isna(max_mutation) else 0.0,
                    'mutation_points': len(e_indices)
                })

            result[event_type] = pd.DataFrame(event_list)

        return result

    def correlate_events(
        self,
        anomaly_events: pd.DataFrame,
        weather_events: Dict[str, pd.DataFrame],
        calendar_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if len(anomaly_events) == 0:
            return anomaly_events

        result = anomaly_events.copy()
        result['has_weather'] = False
        result['has_calendar'] = False
        result['weather_event_types'] = ''
        result['calendar_info'] = ''
        result['related_weather_events'] = ''

        all_weather = []
        for wtype, wdf in weather_events.items():
            if len(wdf) > 0:
                wdf_cp = wdf.copy()
                wdf_cp['event_type'] = wtype
                all_weather.append(wdf_cp)

        if all_weather:
            all_weather_df = pd.concat(all_weather, ignore_index=True)
        else:
            all_weather_df = pd.DataFrame(columns=['start_time', 'end_time', 'event_type'])

        window = timedelta(hours=self.correlation_window_hours)

        for idx, row in result.iterrows():
            anomaly_start = row['start_time']
            window_start = anomaly_start - window
            window_end = anomaly_start + window

            if len(all_weather_df) > 0:
                matched = all_weather_df[
                    ~((all_weather_df['end_time'] < window_start) |
                      (all_weather_df['start_time'] > window_end))
                ]
                if len(matched) > 0:
                    result.at[idx, 'has_weather'] = True
                    types = matched['event_type'].unique().tolist()
                    type_names = []
                    for t in types:
                        type_names.append('气温突变' if t == 'temperature' else '辐照突变')
                    result.at[idx, 'weather_event_types'] = '、'.join(type_names)

                    weather_details = []
                    for _, wrow in matched.iterrows():
                        wtype_cn = '气温突变' if wrow['event_type'] == 'temperature' else '辐照突变'
                        mag = wrow.get('mutation_magnitude', 0)
                        if wrow['event_type'] == 'temperature':
                            mag_str = f"{mag:.1f}°C"
                        else:
                            mag_str = f"{mag:.0f} W/m²"
                        weather_details.append(
                            f"{wtype_cn}: {wrow['start_time'].strftime('%H:%M')}-{wrow['end_time'].strftime('%H:%M')}, 突变幅度{mag_str}"
                        )
                    result.at[idx, 'related_weather_events'] = '|'.join(weather_details)

            if calendar_df is not None and len(calendar_df) > 0:
                cal = calendar_df.copy()
                if 'date' in cal.columns:
                    cal['date'] = pd.to_datetime(cal['date']).dt.date
                event_date = anomaly_start.date()

                matched_cal = cal[cal['date'] == event_date]
                if len(matched_cal) > 0:
                    cal_row = matched_cal.iloc[0]
                    is_holiday = False
                    is_special = False
                    info_parts = []

                    if 'is_holiday' in cal_row:
                        is_holiday = bool(cal_row['is_holiday'])
                    if 'day_type' in cal_row:
                        dt = str(cal_row['day_type'])
                        if dt in ['节假日', '假期']:
                            is_holiday = True
                            info_parts.append(dt)
                        elif dt:
                            info_parts.append(dt)
                    if 'special_event' in cal_row:
                        se = str(cal_row['special_event']).strip()
                        if se and se not in ['nan', 'None', '']:
                            is_special = True
                            info_parts.append(f"特殊事件:{se}")

                    if is_holiday or is_special:
                        result.at[idx, 'has_calendar'] = True
                        result.at[idx, 'calendar_info'] = '、'.join(info_parts) if info_parts else '特殊日'

        def classify(row):
            if row['has_weather'] and row['has_calendar']:
                return '复合关联(气象+日历)'
            elif row['has_weather']:
                return '气象关联'
            elif row['has_calendar']:
                return '日历关联'
            else:
                return '未知原因'

        result['correlation_type'] = result.apply(classify, axis=1)
        return result

    def run_full_analysis(
        self,
        load_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        anomaly_df = self.detect_anomalies(load_df)
        anomaly_events = self.aggregate_anomaly_events(anomaly_df)
        weather_df = self.detect_weather_mutations(load_df)
        weather_events = self.aggregate_weather_events(weather_df)
        correlated_events = self.correlate_events(anomaly_events, weather_events, calendar_df)

        correlation_stats = None
        if len(correlated_events) > 0:
            counts = correlated_events['correlation_type'].value_counts()
            total = len(correlated_events)
            correlation_stats = pd.DataFrame({
                '关联类型': counts.index,
                '事件数量': counts.values,
                '占比': counts.values / total
            })

        monthly_stats = None
        if len(correlated_events) > 0:
            events_cp = correlated_events.copy()
            events_cp['month'] = pd.to_datetime(events_cp['start_time']).dt.to_period('M')
            monthly_counts = events_cp.groupby('month').size().reset_index(name='异常事件数')
            monthly_counts['month'] = monthly_counts['month'].astype(str)
            monthly_stats = monthly_counts

        return {
            'anomaly_points': anomaly_df,
            'anomaly_events': correlated_events,
            'weather_mutations': weather_df,
            'weather_events': weather_events,
            'correlation_stats': correlation_stats,
            'monthly_stats': monthly_stats
        }
