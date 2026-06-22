import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter, HourLocator

from .utils import safe_divide, get_season


class PeakValleyAnalyzer:
    """峰谷分析类，用于提取和分析电力负荷的峰谷特征"""

    def __init__(
        self,
        target_col: str = 'load',
        datetime_col: str = 'datetime',
        peak_threshold_ratio: float = 1.2
    ) -> None:
        """
        初始化峰谷分析器

        Args:
            target_col: 目标列名
            datetime_col: 时间列名
            peak_threshold_ratio: 尖峰阈值比例，默认1.2（超过日均值120%）
        """
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.peak_threshold_ratio = peak_threshold_ratio
        self.daily_features_: Optional[pd.DataFrame] = None
        self.monthly_features_: Optional[pd.DataFrame] = None

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证和预处理数据"""
        df = df.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.sort_values(self.datetime_col).reset_index(drop=True)

        if not pd.api.types.is_numeric_dtype(df[self.target_col]):
            raise ValueError(f"目标列 {self.target_col} 必须是数值型")

        return df

    def extract_daily_peak_valley(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取日峰值/谷值时刻

        Args:
            df: 输入数据

        Returns:
            每日峰谷特征DataFrame
        """
        df = self._validate_data(df)
        df['date'] = df[self.datetime_col].dt.date

        daily_stats = df.groupby('date').agg(
            peak_value=(self.target_col, 'max'),
            valley_value=(self.target_col, 'min'),
            mean_value=(self.target_col, 'mean'),
            total_load=(self.target_col, 'sum')
        ).reset_index()

        peak_times = df.loc[
            df.groupby('date')[self.target_col].idxmax()
        ][['date', self.datetime_col, self.target_col]].rename(
            columns={self.datetime_col: 'peak_time', self.target_col: 'peak_value_check'}
        )

        valley_times = df.loc[
            df.groupby('date')[self.target_col].idxmin()
        ][['date', self.datetime_col, self.target_col]].rename(
            columns={self.datetime_col: 'valley_time', self.target_col: 'valley_value_check'}
        )

        daily_features = pd.merge(daily_stats, peak_times, on='date', how='left')
        daily_features = pd.merge(daily_features, valley_times, on='date', how='left')

        daily_features['peak_hour'] = daily_features['peak_time'].dt.hour
        daily_features['valley_hour'] = daily_features['valley_time'].dt.hour

        daily_features['peak_valley_diff'] = daily_features['peak_value'] - daily_features['valley_value']
        daily_features['daily_load_rate'] = daily_features['mean_value'] / daily_features['peak_value']
        daily_features['load_variation_rate'] = daily_features['peak_valley_diff'] / daily_features['mean_value']

        self.daily_features_ = daily_features
        return daily_features

    def calculate_peak_duration(
        self,
        df: pd.DataFrame,
        threshold_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        计算尖峰持续时间（超过日均值指定比例的连续时段）

        Args:
            df: 输入数据
            threshold_ratio: 阈值比例，默认使用初始化时的1.2

        Returns:
            包含尖峰持续时间的每日特征DataFrame
        """
        if threshold_ratio is None:
            threshold_ratio = self.peak_threshold_ratio

        df = self._validate_data(df)
        df['date'] = df[self.datetime_col].dt.date

        if self.daily_features_ is None:
            self.extract_daily_peak_valley(df)

        daily_means = df.groupby('date')[self.target_col].mean().reset_index()
        daily_means.columns = ['date', 'daily_mean']
        df = pd.merge(df, daily_means, on='date', how='left')

        df['is_peak'] = df[self.target_col] > (df['daily_mean'] * threshold_ratio)

        peak_durations = []
        peak_segments = []

        for date, group in df.groupby('date'):
            is_peak = group['is_peak'].values
            times = group[self.datetime_col].values
            values = group[self.target_col].values

            segments = []
            current_start = None
            current_duration = 0

            for i, (peak, t, v) in enumerate(zip(is_peak, times, values)):
                if peak:
                    if current_start is None:
                        current_start = t
                    current_duration += 1
                else:
                    if current_start is not None:
                        segments.append({
                            'date': date,
                            'start_time': current_start,
                            'end_time': times[i - 1],
                            'duration_points': current_duration,
                            'duration_hours': current_duration * 0.25,
                            'avg_value': np.mean(values[i - current_duration:i])
                        })
                        current_start = None
                        current_duration = 0

            if current_start is not None:
                segments.append({
                    'date': date,
                    'start_time': current_start,
                    'end_time': times[-1],
                    'duration_points': current_duration,
                    'duration_hours': current_duration * 0.25,
                    'avg_value': np.mean(values[-current_duration:])
                })

            peak_segments.extend(segments)

            total_duration = sum(s['duration_hours'] for s in segments)
            max_duration = max((s['duration_hours'] for s in segments), default=0.0)
            peak_count = len(segments)

            peak_durations.append({
                'date': date,
                'peak_duration_hours': total_duration,
                'max_continuous_peak_hours': max_duration,
                'peak_segment_count': peak_count
            })

        duration_df = pd.DataFrame(peak_durations)
        segments_df = pd.DataFrame(peak_segments)

        if self.daily_features_ is not None:
            self.daily_features_ = pd.merge(
                self.daily_features_,
                duration_df,
                on='date',
                how='left'
            )

        self.peak_segments_ = segments_df
        return self.daily_features_

    def get_daily_load_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算日负荷率（平均/峰值）

        Args:
            df: 输入数据

        Returns:
            包含日负荷率的DataFrame
        """
        if self.daily_features_ is None:
            self.extract_daily_peak_valley(df)

        result = self.daily_features_[['date', 'mean_value', 'peak_value', 'daily_load_rate']].copy()
        result['daily_load_rate'] = result['daily_load_rate'].apply(
            lambda x: safe_divide(result.loc[result.index[result['date'] == x.name], 'mean_value'].iloc[0],
                                  result.loc[result.index[result['date'] == x.name], 'peak_value'].iloc[0])
        )
        return result

    def get_peak_valley_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算峰谷差

        Args:
            df: 输入数据

        Returns:
            包含峰谷差的DataFrame
        """
        if self.daily_features_ is None:
            self.extract_daily_peak_valley(df)

        return self.daily_features_[['date', 'peak_value', 'valley_value', 'peak_valley_diff']].copy()

    def aggregate_monthly_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        月度峰谷特征汇总和趋势分析

        Args:
            df: 输入数据，None时使用已计算的每日特征

        Returns:
            月度特征汇总DataFrame
        """
        if df is not None and self.daily_features_ is None:
            self.extract_daily_peak_valley(df)
            self.calculate_peak_duration(df)

        if self.daily_features_ is None:
            raise ValueError("请先提供数据或计算每日特征")

        daily = self.daily_features_.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        daily['year_month'] = daily['date'].dt.to_period('M')
        daily['month'] = daily['date'].dt.month
        daily['season'] = daily['month'].apply(get_season)

        monthly = daily.groupby('year_month').agg(
            avg_peak_value=('peak_value', 'mean'),
            avg_valley_value=('valley_value', 'mean'),
            avg_peak_valley_diff=('peak_valley_diff', 'mean'),
            avg_daily_load_rate=('daily_load_rate', 'mean'),
            avg_peak_duration_hours=('peak_duration_hours', 'mean'),
            max_peak_value=('peak_value', 'max'),
            min_valley_value=('valley_value', 'min'),
            days_count=('date', 'count')
        ).reset_index()

        monthly['peak_trend'] = monthly['avg_peak_value'].pct_change() * 100
        monthly['valley_trend'] = monthly['avg_valley_value'].pct_change() * 100
        monthly['load_rate_trend'] = monthly['avg_daily_load_rate'].pct_change() * 100

        peak_hour_mode = daily.groupby('year_month')['peak_hour'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else np.nan
        ).reset_index(name='typical_peak_hour')

        valley_hour_mode = daily.groupby('year_month')['valley_hour'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else np.nan
        ).reset_index(name='typical_valley_hour')

        monthly = pd.merge(monthly, peak_hour_mode, on='year_month', how='left')
        monthly = pd.merge(monthly, valley_hour_mode, on='year_month', how='left')

        season_map = daily.groupby('year_month')['season'].first().reset_index()
        monthly = pd.merge(monthly, season_map, on='year_month', how='left')

        self.monthly_features_ = monthly
        return monthly

    def analyze_trend(
        self,
        df: Optional[pd.DataFrame] = None,
        window: int = 7
    ) -> Dict[str, Any]:
        """
        趋势分析

        Args:
            df: 输入数据
            window: 移动平均窗口大小

        Returns:
            趋势分析结果字典
        """
        if df is not None:
            self.extract_daily_peak_valley(df)
            self.calculate_peak_duration(df)

        if self.daily_features_ is None:
            raise ValueError("请先提供数据或计算每日特征")

        daily = self.daily_features_.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date')

        daily['peak_ma'] = daily['peak_value'].rolling(window=window).mean()
        daily['valley_ma'] = daily['valley_value'].rolling(window=window).mean()
        daily['load_rate_ma'] = daily['daily_load_rate'].rolling(window=window).mean()

        overall_trend = {
            'peak_value': {
                'start': daily['peak_value'].iloc[0],
                'end': daily['peak_value'].iloc[-1],
                'change_pct': ((daily['peak_value'].iloc[-1] - daily['peak_value'].iloc[0]) /
                               daily['peak_value'].iloc[0] * 100) if daily['peak_value'].iloc[0] != 0 else 0
            },
            'valley_value': {
                'start': daily['valley_value'].iloc[0],
                'end': daily['valley_value'].iloc[-1],
                'change_pct': ((daily['valley_value'].iloc[-1] - daily['valley_value'].iloc[0]) /
                               daily['valley_value'].iloc[0] * 100) if daily['valley_value'].iloc[0] != 0 else 0
            },
            'daily_load_rate': {
                'start': daily['daily_load_rate'].iloc[0],
                'end': daily['daily_load_rate'].iloc[-1],
                'change_pct': ((daily['daily_load_rate'].iloc[-1] - daily['daily_load_rate'].iloc[0]) /
                               daily['daily_load_rate'].iloc[0] * 100) if daily['daily_load_rate'].iloc[0] != 0 else 0
            }
        }

        return {
            'moving_average': daily[['date', 'peak_ma', 'valley_ma', 'load_rate_ma']].dropna(),
            'overall_trend': overall_trend,
            'statistics': {
                'avg_peak': daily['peak_value'].mean(),
                'avg_valley': daily['valley_value'].mean(),
                'avg_load_rate': daily['daily_load_rate'].mean(),
                'max_peak': daily['peak_value'].max(),
                'min_valley': daily['valley_value'].min()
            }
        }

    def plot_peak_valley_markers(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8),
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        峰谷时段可视化标记

        Args:
            df: 输入数据
            start_date: 开始日期
            end_date: 结束日期
            figsize: 图表大小
            show: 是否显示图表

        Returns:
            matplotlib Figure对象
        """
        df = self._validate_data(df)

        if start_date:
            df = df[df[self.datetime_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[self.datetime_col] <= pd.to_datetime(end_date)]

        if len(df) == 0:
            raise ValueError("指定日期范围内无数据")

        if self.daily_features_ is None:
            self.extract_daily_peak_valley(df)

        if not hasattr(self, 'peak_segments_') or self.peak_segments_ is None:
            self.calculate_peak_duration(df)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(df[self.datetime_col], df[self.target_col], label='负荷曲线', color='blue', linewidth=1)

        peak_mask = df[self.datetime_col].isin(self.daily_features_['peak_time'])
        valley_mask = df[self.datetime_col].isin(self.daily_features_['valley_time'])

        ax.scatter(
            df.loc[peak_mask, self.datetime_col],
            df.loc[peak_mask, self.target_col],
            color='red', s=100, zorder=5, label='日峰值', marker='^'
        )
        ax.scatter(
            df.loc[valley_mask, self.datetime_col],
            df.loc[valley_mask, self.target_col],
            color='green', s=100, zorder=5, label='日谷值', marker='v'
        )

        for _, seg in self.peak_segments_.iterrows():
            if start_date and seg['end_time'] < pd.to_datetime(start_date):
                continue
            if end_date and seg['start_time'] > pd.to_datetime(end_date):
                continue
            ax.axvspan(
                seg['start_time'], seg['end_time'],
                alpha=0.3, color='orange', label='_nolegend_'
            )

        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('负荷', fontsize=12)
        ax.set_title('负荷曲线峰谷时段标记', fontsize=14, fontweight='bold')

        legend_elements = [
            Patch(facecolor='orange', alpha=0.3, label='尖峰时段'),
            plt.Line2D([0], [0], color='blue', linewidth=1, label='负荷曲线'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='日峰值'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='green', markersize=10, label='日谷值')
        ]
        ax.legend(handles=legend_elements, loc='best')

        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(HourLocator(interval=6))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()
        return fig

    def get_peak_segments(self) -> pd.DataFrame:
        """
        获取尖峰时段分段信息

        Returns:
            尖峰分段DataFrame
        """
        if not hasattr(self, 'peak_segments_') or self.peak_segments_ is None:
            raise ValueError("请先调用 calculate_peak_duration 方法计算尖峰持续时间")
        return self.peak_segments_

    def summary(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        生成峰谷分析总览

        Args:
            df: 输入数据

        Returns:
            分析总览字典
        """
        if df is not None:
            self.extract_daily_peak_valley(df)
            self.calculate_peak_duration(df)
            self.aggregate_monthly_features()

        if self.daily_features_ is None:
            raise ValueError("请先提供数据进行分析")

        daily = self.daily_features_

        return {
            'data_range': {
                'start_date': str(daily['date'].min()),
                'end_date': str(daily['date'].max()),
                'days_count': len(daily)
            },
            'peak_statistics': {
                'avg_peak': daily['peak_value'].mean(),
                'max_peak': daily['peak_value'].max(),
                'min_peak': daily['peak_value'].min(),
                'typical_peak_hour': daily['peak_hour'].mode().iloc[0] if len(daily) > 0 else None
            },
            'valley_statistics': {
                'avg_valley': daily['valley_value'].mean(),
                'max_valley': daily['valley_value'].max(),
                'min_valley': daily['valley_value'].min(),
                'typical_valley_hour': daily['valley_hour'].mode().iloc[0] if len(daily) > 0 else None
            },
            'load_rate_statistics': {
                'avg_load_rate': daily['daily_load_rate'].mean(),
                'min_load_rate': daily['daily_load_rate'].min(),
                'max_load_rate': daily['daily_load_rate'].max()
            },
            'peak_duration_statistics': {
                'avg_peak_duration': daily['peak_duration_hours'].mean(),
                'max_peak_duration': daily['peak_duration_hours'].max(),
                'avg_continuous_peak': daily['max_continuous_peak_hours'].mean()
            } if 'peak_duration_hours' in daily.columns else None
        }
