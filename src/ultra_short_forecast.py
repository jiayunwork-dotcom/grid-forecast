import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque

from .utils import mape, rmse, datetime_features, sliding_window


class UltraShortTermForecaster:
    """超短期预测模型，支持滑动窗口预测和在线增量学习"""

    def __init__(
        self,
        target_col: str = 'load',
        feature_cols: Optional[List[str]] = None,
        input_window_size: int = 8,
        forecast_horizon: int = 16,
        freq: str = '15T',
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化超短期预测器

        Args:
            target_col: 目标列名
            feature_cols: 特征列名列表
            input_window_size: 输入窗口大小（点数），默认8个点=2小时（15分钟间隔）
            forecast_horizon: 预测步长（点数），默认16个点=4小时（15分钟间隔）
            freq: 数据频率
            params: LightGBM参数字典
        """
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.input_window_size = input_window_size
        self.forecast_horizon = forecast_horizon
        self.freq = freq
        self.model: Optional[lgb.Booster] = None
        self.scaler: Optional[StandardScaler] = None
        self.data_buffer: deque = deque(maxlen=input_window_size)
        self.last_update_time: Optional[pd.Timestamp] = None

        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.params = params or default_params

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和目标数据"""
        if self.feature_cols is None:
            self.feature_cols = [
                col for col in df.columns
                if col != self.target_col and pd.api.types.is_numeric_dtype(df[col])
            ]

        X = df[self.feature_cols].values.astype(np.float64)
        y = df[self.target_col].values.astype(np.float64)
        return X, y

    def _build_sliding_window_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建滑动窗口特征

        Args:
            X: 特征数据，形状 (n_samples, n_features)
            y: 目标数据，形状 (n_samples,)

        Returns:
            X_window: 窗口特征，形状 (n_windows, input_window_size * n_features)
            y_window: 多步预测目标，形状 (n_windows, forecast_horizon)
        """
        n_samples = len(y)
        if n_samples < self.input_window_size + self.forecast_horizon:
            raise ValueError(
                f"数据量不足，需要至少 {self.input_window_size + self.forecast_horizon} 个样本"
            )

        n_windows = n_samples - self.input_window_size - self.forecast_horizon + 1
        n_features = X.shape[1]

        X_window = np.zeros((n_windows, self.input_window_size * n_features))
        y_window = np.zeros((n_windows, self.forecast_horizon))

        for i in range(n_windows):
            X_window[i] = X[i:i + self.input_window_size].flatten()
            y_window[i] = y[i + self.input_window_size:i + self.input_window_size + self.forecast_horizon]

        return X_window, y_window

    def build_features(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        特征工程接口

        Args:
            df: 原始数据
            datetime_col: 时间列名

        Returns:
            包含特征的DataFrame
        """
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

        dt_features = datetime_features(pd.DatetimeIndex(df[datetime_col]))
        df = pd.concat([df, dt_features.reset_index(drop=True)], axis=1)

        df['load_lag_1'] = df[self.target_col].shift(1)
        df['load_lag_2'] = df[self.target_col].shift(2)
        df['load_rolling_mean_4'] = df[self.target_col].rolling(window=4).mean()
        df['load_rolling_std_4'] = df[self.target_col].rolling(window=4).std()

        df = df.bfill().ffill()

        if self.feature_cols is None:
            self.feature_cols = [
                col for col in df.columns
                if col != self.target_col and col != datetime_col
                and pd.api.types.is_numeric_dtype(df[col])
            ]

        return df

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ) -> 'UltraShortTermForecaster':
        """
        训练模型

        Args:
            train_df: 训练数据集
            val_df: 验证数据集
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 提前停止轮数

        Returns:
            训练后的实例
        """
        X_train, y_train = self._prepare_features(train_df)
        X_train_window, y_train_window = self._build_sliding_window_features(X_train, y_train)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_window)

        valid_sets = []
        valid_names = []

        if val_df is not None:
            X_val, y_val = self._prepare_features(val_df)
            X_val_window, y_val_window = self._build_sliding_window_features(X_val, y_val)
            X_val_scaled = self.scaler.transform(X_val_window)
            lgb_val = lgb.Dataset(X_val_scaled, label=y_val_window)
            valid_sets.append(lgb_val)
            valid_names.append('val')

        lgb_train = lgb.Dataset(X_train_scaled, label=y_train_window)
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets if valid_sets else None,
            valid_names=valid_names if valid_names else None,
            callbacks=[lgb.early_stopping(early_stopping_rounds)] if valid_sets else None
        )

        if len(train_df) >= self.input_window_size:
            recent_data = train_df.tail(self.input_window_size)[self.target_col].values
            self.data_buffer.extend(recent_data.tolist())
            self.last_update_time = train_df.iloc[-1]['datetime']

        return self

    def predict(
        self,
        df: pd.DataFrame,
        with_interval: bool = False,
        interval_range: float = 0.2
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        滑动窗口预测：使用最近2小时数据预测未来4小时16个点

        Args:
            df: 输入数据，需包含最近input_window_size个点
            with_interval: 是否返回预测区间
            interval_range: 预测区间上下浮动比例，默认20%

        Returns:
            预测值数组或包含预测区间的DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        if len(df) < self.input_window_size:
            raise ValueError(
                f"输入数据不足，需要至少 {self.input_window_size} 个样本"
            )

        recent_df = df.tail(self.input_window_size).copy()
        X, _ = self._prepare_features(recent_df)
        X_flat = X.flatten().reshape(1, -1)
        X_scaled = self.scaler.transform(X_flat)

        predictions = self.model.predict(X_scaled).flatten()

        if with_interval:
            lower_bound = predictions * (1 - interval_range)
            upper_bound = predictions * (1 + interval_range)
            return pd.DataFrame({
                'prediction': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })

        return predictions

    def predict_with_interval(
        self,
        df: pd.DataFrame,
        interval_range: float = 0.2
    ) -> pd.DataFrame:
        """
        带预测区间的预测（上下浮动20%）

        Args:
            df: 输入数据
            interval_range: 区间范围比例，默认0.2即20%

        Returns:
            包含预测值、上下界的DataFrame
        """
        return self.predict(df, with_interval=True, interval_range=interval_range)

    def update(self, new_df: pd.DataFrame) -> None:
        """
        在线增量学习：新数据到达后更新模型

        Args:
            new_df: 新数据
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法进行增量更新")

        new_df = new_df.copy()
        new_df['datetime'] = pd.to_datetime(new_df['datetime'])
        new_df = new_df.sort_values('datetime').reset_index(drop=True)

        X_new, y_new = self._prepare_features(new_df)
        X_new_window, y_new_window = self._build_sliding_window_features(X_new, y_new)

        X_new_scaled = self.scaler.transform(X_new_window)

        lgb_new = lgb.Dataset(X_new_scaled, label=y_new_window)
        self.model = lgb.train(
            self.params,
            lgb_new,
            num_boost_round=50,
            init_model=self.model,
            keep_training_booster=True
        )

        new_load_values = new_df[self.target_col].values.tolist()
        self.data_buffer.extend(new_load_values)
        self.last_update_time = new_df.iloc[-1]['datetime']

    def rolling_predict(
        self,
        df: pd.DataFrame,
        step_size: int = 1,
        with_interval: bool = False
    ) -> pd.DataFrame:
        """
        滚动预测：在整个数据集上进行滑动窗口预测

        Args:
            df: 输入数据
            step_size: 滚动步长
            with_interval: 是否返回预测区间

        Returns:
            预测结果DataFrame，包含时间索引
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        results = []
        n_samples = len(df)
        start_idx = self.input_window_size

        for i in range(start_idx, n_samples - self.forecast_horizon + 1, step_size):
            window_df = df.iloc[i - self.input_window_size:i]
            forecast_start_time = df.iloc[i]['datetime']
            forecast_times = pd.date_range(
                start=forecast_start_time,
                periods=self.forecast_horizon,
                freq=self.freq
            )

            if with_interval:
                pred_df = self.predict(window_df, with_interval=True)
                pred_df['datetime'] = forecast_times
                pred_df['origin_time'] = forecast_start_time
                pred_df['horizon'] = np.arange(self.forecast_horizon)
                results.append(pred_df)
            else:
                predictions = self.predict(window_df)
                results.append(pd.DataFrame({
                    'datetime': forecast_times,
                    'origin_time': forecast_start_time,
                    'horizon': np.arange(self.forecast_horizon),
                    'prediction': predictions
                }))

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算预测评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            包含MAPE和RMSE的字典
        """
        return {
            'mape': mape(y_true, y_pred),
            'rmse': rmse(y_true, y_pred)
        }

    def get_buffer_data(self) -> np.ndarray:
        """
        获取当前数据缓冲区的数据

        Returns:
            缓冲区中的数据
        """
        return np.array(self.data_buffer)

    def get_last_update_time(self) -> Optional[pd.Timestamp]:
        """
        获取上次模型更新时间

        Returns:
            上次更新时间
        """
        return self.last_update_time
