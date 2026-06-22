import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Literal
from collections import Counter
import json
import os
from scipy import stats

from .utils import mape, rmse, safe_divide


ErrorCategory = Literal['突发天气', '节假日效应', '特殊事件', '数据质量问题']


class ErrorAnalyzer:
    """误差分析类，用于分析预测误差的统计特征和归因"""

    def __init__(
        self,
        target_col: str = 'load',
        prediction_col: str = 'prediction',
        datetime_col: str = 'datetime',
        large_error_threshold: float = 8.0
    ) -> None:
        """
        初始化误差分析器

        Args:
            target_col: 真实值列名
            prediction_col: 预测值列名
            datetime_col: 时间列名
            large_error_threshold: 大误差阈值（MAPE百分比），默认8%
        """
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.datetime_col = datetime_col
        self.large_error_threshold = large_error_threshold
        self.daily_errors_: Optional[pd.DataFrame] = None
        self.error_attributions_: Optional[pd.DataFrame] = None
        self.error_distribution_: Optional[Dict[str, Any]] = None

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证和预处理数据"""
        df = df.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.sort_values(self.datetime_col).reset_index(drop=True)

        required_cols = [self.target_col, self.prediction_col, self.datetime_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

        return df

    def _calculate_point_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算逐点误差"""
        df = df.copy()
        df['error'] = df[self.target_col] - df[self.prediction_col]
        df['abs_error'] = df['error'].abs()
        df['ape'] = np.where(
            df[self.target_col] != 0,
            (df['abs_error'] / df[self.target_col].abs()) * 100,
            0.0
        )
        return df

    def calculate_daily_mape(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按日统计MAPE

        Args:
            df: 包含真实值和预测值的数据

        Returns:
            每日误差统计DataFrame
        """
        df = self._validate_data(df)
        df = self._calculate_point_errors(df)
        df['date'] = df[self.datetime_col].dt.date

        daily_stats = df.groupby('date').agg(
            samples_count=(self.target_col, 'count'),
            avg_abs_error=('abs_error', 'mean'),
            rmse=(self.target_col, lambda x: rmse(x.values, df.loc[x.index, self.prediction_col].values)),
            mape=(self.target_col, lambda x: mape(x.values, df.loc[x.index, self.prediction_col].values))
        ).reset_index()

        daily_stats['is_large_error'] = daily_stats['mape'] > self.large_error_threshold

        self.daily_errors_ = daily_stats
        return daily_stats

    def identify_large_error_days(
        self,
        df: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        识别大误差日（MAPE超过阈值）

        Args:
            df: 输入数据，None时使用已计算的每日误差
            threshold: 大误差阈值，默认使用初始化时的8%

        Returns:
            大误差日DataFrame
        """
        if threshold is None:
            threshold = self.large_error_threshold

        if df is not None:
            self.calculate_daily_mape(df)

        if self.daily_errors_ is None:
            raise ValueError("请先提供数据或调用 calculate_daily_mape 方法")

        large_errors = self.daily_errors_[
            self.daily_errors_['mape'] > threshold
        ].copy().sort_values('mape', ascending=False)

        return large_errors

    def attribute_error(
        self,
        date: Union[str, pd.Timestamp],
        category: ErrorCategory,
        description: Optional[str] = None,
        confidence: float = 1.0
    ) -> None:
        """
        标注误差归因

        Args:
            date: 日期
            category: 误差类别（突发天气、节假日效应、特殊事件、数据质量问题）
            description: 详细描述
            confidence: 置信度 0-1
        """
        valid_categories: List[ErrorCategory] = ['突发天气', '节假日效应', '特殊事件', '数据质量问题']
        if category not in valid_categories:
            raise ValueError(f"无效的误差类别，必须是: {valid_categories}")

        date = pd.to_datetime(date).date()

        new_row = pd.DataFrame([{
            'date': date,
            'category': category,
            'description': description,
            'confidence': confidence,
            'created_at': pd.Timestamp.now()
        }])

        if self.error_attributions_ is None or self.error_attributions_.empty:
            self.error_attributions_ = new_row
        else:
            existing_idx = self.error_attributions_[self.error_attributions_['date'] == date].index
            if len(existing_idx) > 0:
                self.error_attributions_.loc[existing_idx[0]] = new_row.iloc[0]
            else:
                self.error_attributions_ = pd.concat(
                    [self.error_attributions_, new_row],
                    ignore_index=True
                )

    def batch_attribute_errors(
        self,
        attributions: List[Dict[str, Any]]
    ) -> None:
        """
        批量标注误差归因

        Args:
            attributions: 归因列表，每个元素包含date, category, description, confidence
        """
        for attr in attributions:
            self.attribute_error(
                date=attr['date'],
                category=attr['category'],
                description=attr.get('description'),
                confidence=attr.get('confidence', 1.0)
            )

    def get_error_attributions(self) -> pd.DataFrame:
        """
        获取所有误差归因

        Returns:
            归因DataFrame
        """
        if self.error_attributions_ is None:
            return pd.DataFrame(columns=[
                'date', 'category', 'description', 'confidence', 'created_at'
            ])
        return self.error_attributions_.copy()

    def analyze_error_distribution(
        self,
        df: Optional[pd.DataFrame] = None,
        bins: int = 20
    ) -> Dict[str, Any]:
        """
        误差分布统计

        Args:
            df: 输入数据
            bins: 直方图箱数

        Returns:
            误差分布统计字典
        """
        if df is None and self.daily_errors_ is None:
            raise ValueError("请先提供数据")

        if df is not None:
            df = self._validate_data(df)
            df = self._calculate_point_errors(df)

            errors = df['error'].dropna().values
            apes = df['ape'].dropna().values
        else:
            errors = None
            apes = self.daily_errors_['mape'].values

        if errors is not None and len(errors) > 0:
            hist, bin_edges = np.histogram(errors, bins=bins)
            distribution = {
                'point_errors': {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'median': float(np.median(errors)),
                    'min': float(np.min(errors)),
                    'max': float(np.max(errors)),
                    'skewness': float(stats.skew(errors)),
                    'kurtosis': float(stats.kurtosis(errors)),
                    'histogram': {
                        'counts': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    },
                    'percentiles': {
                        '5': float(np.percentile(errors, 5)),
                        '25': float(np.percentile(errors, 25)),
                        '50': float(np.percentile(errors, 50)),
                        '75': float(np.percentile(errors, 75)),
                        '95': float(np.percentile(errors, 95))
                    }
                }
            }
        else:
            distribution = {}

        if apes is not None and len(apes) > 0:
            ape_hist, ape_bin_edges = np.histogram(apes, bins=bins)
            distribution['mape'] = {
                'mean': float(np.mean(apes)),
                'std': float(np.std(apes)),
                'median': float(np.median(apes)),
                'min': float(np.min(apes)),
                'max': float(np.max(apes)),
                'histogram': {
                    'counts': ape_hist.tolist(),
                    'bin_edges': ape_bin_edges.tolist()
                },
                'below_5_pct': float(np.sum(apes < 5) / len(apes) * 100),
                '5_to_8_pct': float(np.sum((apes >= 5) & (apes < 8)) / len(apes) * 100),
                'above_8_pct': float(np.sum(apes >= 8) / len(apes) * 100)
            }

        self.error_distribution_ = distribution
        return distribution

    def analyze_temperature_correlation(
        self,
        df: pd.DataFrame,
        temp_col: str = 'temperature',
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        误差与气温相关性分析

        Args:
            df: 输入数据，需包含气温列
            temp_col: 气温列名
            method: 相关系数方法，'pearson'或'spearman'

        Returns:
            相关性分析结果
        """
        if temp_col not in df.columns:
            raise ValueError(f"数据中缺少气温列: {temp_col}")

        df = self._validate_data(df)
        df = self._calculate_point_errors(df)

        valid_data = df[[temp_col, 'error', 'ape', self.target_col]].dropna()

        if len(valid_data) < 2:
            raise ValueError("有效样本不足，无法计算相关性")

        if method == 'pearson':
            corr_func = stats.pearsonr
        elif method == 'spearman':
            corr_func = stats.spearmanr
        else:
            raise ValueError(f"不支持的相关性方法: {method}")

        temp_error_corr, temp_error_p = corr_func(
            valid_data[temp_col].values,
            valid_data['error'].values
        )
        temp_ape_corr, temp_ape_p = corr_func(
            valid_data[temp_col].values,
            valid_data['ape'].values
        )
        temp_load_corr, temp_load_p = corr_func(
            valid_data[temp_col].values,
            valid_data[self.target_col].values
        )

        valid_data['temp_bin'] = pd.cut(valid_data[temp_col], bins=10)
        bin_stats = valid_data.groupby('temp_bin', observed=False).agg(
            temp_mean=(temp_col, 'mean'),
            avg_error=('error', 'mean'),
            avg_ape=('ape', 'mean'),
            count=(temp_col, 'count')
        ).reset_index()
        bin_stats['temp_bin'] = bin_stats['temp_bin'].astype(str)

        return {
            'method': method,
            'correlations': {
                'temperature_vs_error': {
                    'correlation': float(temp_error_corr),
                    'p_value': float(temp_error_p)
                },
                'temperature_vs_ape': {
                    'correlation': float(temp_ape_corr),
                    'p_value': float(temp_ape_p)
                },
                'temperature_vs_load': {
                    'correlation': float(temp_load_corr),
                    'p_value': float(temp_load_p)
                }
            },
            'bin_statistics': bin_stats.to_dict('records'),
            'sample_count': len(valid_data)
        }

    def save_attributions(self, filepath: str) -> None:
        """
        保存归因数据到文件

        Args:
            filepath: 保存路径（JSON格式）
        """
        if self.error_attributions_ is None:
            data = []
        else:
            data = self.error_attributions_.copy()
            data['date'] = data['date'].astype(str)
            data['created_at'] = data['created_at'].astype(str)
            data = data.to_dict('records')

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'version': '1.0',
                'created_at': pd.Timestamp.now().isoformat(),
                'large_error_threshold': self.large_error_threshold,
                'attributions': data
            }, f, ensure_ascii=False, indent=2)

    def load_attributions(self, filepath: str) -> pd.DataFrame:
        """
        从文件读取归因数据

        Args:
            filepath: 文件路径

        Returns:
            归因DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        attributions = pd.DataFrame(data.get('attributions', []))
        if not attributions.empty:
            attributions['date'] = pd.to_datetime(attributions['date']).dt.date
            attributions['created_at'] = pd.to_datetime(attributions['created_at'])

        self.error_attributions_ = attributions
        self.large_error_threshold = data.get('large_error_threshold', self.large_error_threshold)

        return attributions

    def get_attribution_statistics(self) -> Dict[str, Any]:
        """
        获取归因统计信息

        Returns:
            归因统计字典
        """
        if self.error_attributions_ is None or self.error_attributions_.empty:
            return {
                'total_attributions': 0,
                'category_distribution': {},
                'confidence_distribution': {}
            }

        category_counts = Counter(self.error_attributions_['category'])
        confidence_levels = pd.cut(
            self.error_attributions_['confidence'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['低置信', '中置信', '高置信']
        )
        confidence_counts = Counter(confidence_levels.dropna())

        return {
            'total_attributions': len(self.error_attributions_),
            'category_distribution': dict(category_counts),
            'confidence_distribution': dict(confidence_counts),
            'avg_confidence': float(self.error_attributions_['confidence'].mean())
        }

    def merge_with_attributions(
        self,
        daily_errors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        将每日误差与归因数据合并

        Args:
            daily_errors: 每日误差数据，None时使用已计算的数据

        Returns:
            合并后的DataFrame
        """
        if daily_errors is None:
            if self.daily_errors_ is None:
                raise ValueError("请先计算每日误差或提供数据")
            daily_errors = self.daily_errors_

        if self.error_attributions_ is None or self.error_attributions_.empty:
            result = daily_errors.copy()
            result['category'] = None
            result['description'] = None
            result['confidence'] = None
            return result

        result = pd.merge(
            daily_errors,
            self.error_attributions_,
            on='date',
            how='left'
        )
        return result

    def summary(
        self,
        df: Optional[pd.DataFrame] = None,
        temp_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成误差分析总览

        Args:
            df: 输入数据
            temp_col: 气温列名，提供时进行相关性分析

        Returns:
            分析总览字典
        """
        if df is not None:
            self.calculate_daily_mape(df)
            self.analyze_error_distribution(df)

        if self.daily_errors_ is None:
            raise ValueError("请先提供数据进行分析")

        daily = self.daily_errors_
        large_errors = self.identify_large_error_days()

        summary = {
            'data_range': {
                'start_date': str(daily['date'].min()),
                'end_date': str(daily['date'].max()),
                'days_count': len(daily)
            },
            'overall_metrics': {
                'avg_mape': float(daily['mape'].mean()),
                'min_mape': float(daily['mape'].min()),
                'max_mape': float(daily['mape'].max()),
                'median_mape': float(daily['mape'].median()),
                'avg_rmse': float(daily['rmse'].mean())
            },
            'large_error_days': {
                'count': len(large_errors),
                'percentage': safe_divide(len(large_errors), len(daily)) * 100,
                'avg_mape': float(large_errors['mape'].mean()) if len(large_errors) > 0 else 0,
                'max_mape': float(large_errors['mape'].max()) if len(large_errors) > 0 else 0
            },
            'attribution_statistics': self.get_attribution_statistics()
        }

        if temp_col is not None and df is not None and temp_col in df.columns:
            summary['temperature_correlation'] = self.analyze_temperature_correlation(df, temp_col)

        if self.error_distribution_:
            summary['error_distribution'] = self.error_distribution_

        return summary
