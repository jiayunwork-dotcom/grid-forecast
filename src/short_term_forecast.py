import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict, Any, Union, Literal
from collections import OrderedDict

from .utils import mape, rmse, time_series_split, create_sequences, datetime_features


class LightGBMForecaster:
    """LightGBM短期预测模型，支持分位数回归用于预测区间估计"""

    def __init__(
        self,
        target_col: str = 'load',
        feature_cols: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化LightGBM预测器

        Args:
            target_col: 目标列名
            feature_cols: 特征列名列表，None时自动选择数值型特征
            quantiles: 分位数列表用于分位数回归，如[0.1, 0.5, 0.9]，None时使用普通回归
            params: LightGBM参数字典
        """
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models: Dict[float, lgb.Booster] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_importance_: Optional[pd.DataFrame] = None

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

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ) -> 'LightGBMForecaster':
        """
        训练LightGBM模型

        Args:
            train_df: 训练数据集
            val_df: 验证数据集
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 提前停止轮数

        Returns:
            训练后的模型实例
        """
        X_train, y_train = self._prepare_features(train_df)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        valid_sets = []
        valid_names = []

        if val_df is not None:
            X_val, y_val = self._prepare_features(val_df)
            X_val_scaled = self.scaler.transform(X_val)
            lgb_val = lgb.Dataset(X_val_scaled, label=y_val)
            valid_sets.append(lgb_val)
            valid_names.append('val')

        if 0.5 in self.quantiles:
            self.quantiles.remove(0.5)
            self.quantiles.insert(0, 0.5)

        for q in self.quantiles:
            params = self.params.copy()
            if len(self.quantiles) > 1:
                params['objective'] = 'quantile'
                params['alpha'] = q
                params['metric'] = 'quantile'

            lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets if valid_sets else None,
                valid_names=valid_names if valid_names else None,
                callbacks=[lgb.early_stopping(early_stopping_rounds)] if valid_sets else None
            )
            self.models[q] = model

        if 0.5 in self.models:
            importance = self.models[0.5].feature_importance()
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        return self

    def predict(self, df: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        预测

        Args:
            df: 输入数据

        Returns:
            单模型返回预测值数组，多模型返回包含各分位数预测的DataFrame
        """
        X, _ = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        if len(self.models) == 1 and 0.5 in self.models:
            return self.models[0.5].predict(X_scaled)

        predictions = {}
        for q, model in self.models.items():
            predictions[f'q{int(q * 100)}'] = model.predict(X_scaled)

        result = pd.DataFrame(predictions, index=df.index)
        if 'q50' in result.columns:
            result['prediction'] = result['q50']
        return result

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征，None返回全部

        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance_ is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")

        if top_n is not None:
            return self.feature_importance_.head(top_n)
        return self.feature_importance_

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

    def update(self, new_df: pd.DataFrame, keep_n_estimators: Optional[int] = None) -> None:
        """
        增量更新模型

        Args:
            new_df: 新数据
            keep_n_estimators: 保留的迭代次数，None保留全部
        """
        X_new, y_new = self._prepare_features(new_df)
        X_new_scaled = self.scaler.transform(X_new)

        for q, model in self.models.items():
            params = self.params.copy()
            if len(self.models) > 1:
                params['objective'] = 'quantile'
                params['alpha'] = q
                params['metric'] = 'quantile'

            lgb_new = lgb.Dataset(X_new_scaled, label=y_new)
            init_model = model if keep_n_estimators is None else model
            self.models[q] = lgb.train(
                params,
                lgb_new,
                num_boost_round=100,
                init_model=init_model,
                keep_training_booster=True
            )


class LSTMModel(nn.Module):
    """双层LSTM网络结构"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ) -> None:
        """
        初始化LSTM网络

        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout概率
            output_size: 输出维度
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, seq_length, input_size)

        Returns:
            输出张量，形状 (batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class LSTMForecaster:
    """基于PyTorch的LSTM短期预测模型"""

    def __init__(
        self,
        target_col: str = 'load',
        feature_cols: Optional[List[str]] = None,
        seq_length: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: Optional[str] = None
    ) -> None:
        """
        初始化LSTM预测器

        Args:
            target_col: 目标列名
            feature_cols: 特征列名列表
            seq_length: 序列长度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout概率
            learning_rate: 学习率
            batch_size: 批次大小
            device: 运行设备，'cpu'或'cuda'
        """
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: Optional[LSTMModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.input_size: Optional[int] = None

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和目标数据"""
        if self.feature_cols is None:
            self.feature_cols = [
                col for col in df.columns
                if col != self.target_col and pd.api.types.is_numeric_dtype(df[col])
            ]

        X = df[self.feature_cols].values.astype(np.float64)
        y = df[self.target_col].values.astype(np.float64).reshape(-1, 1)
        return X, y

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False
    ) -> DataLoader:
        """创建数据加载器"""
        X_seq, y_seq = create_sequences(X, y, self.seq_length)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> 'LSTMForecaster':
        """
        训练LSTM模型

        Args:
            train_df: 训练数据集
            val_df: 验证数据集
            epochs: 最大训练轮数
            early_stopping_patience: 提前停止耐心值

        Returns:
            训练后的模型实例
        """
        X_train, y_train = self._prepare_features(train_df)
        self.input_size = X_train.shape[1]

        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train)

        train_loader = self._create_dataloader(X_train_scaled, y_train_scaled, shuffle=True)

        val_loader = None
        if val_df is not None:
            X_val, y_val = self._prepare_features(val_df)
            X_val_scaled = self.scaler.transform(X_val)
            y_val_scaled = self.target_scaler.transform(y_val)
            val_loader = self._create_dataloader(X_val_scaled, y_val_scaled, shuffle=False)

        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                val_loss /= len(val_loader.dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.model.load_state_dict(best_model_state)
                    break

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            df: 输入数据

        Returns:
            预测值数组
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        X, y = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        X_seq, _ = create_sequences(X_scaled, np.zeros(len(X)), self.seq_length)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()

        padded_predictions = np.full(len(df), np.nan)
        padded_predictions[self.seq_length:] = predictions
        return padded_predictions

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算预测评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            包含MAPE和RMSE的字典
        """
        mask = ~np.isnan(y_pred)
        return {
            'mape': mape(y_true[mask], y_pred[mask]),
            'rmse': rmse(y_true[mask], y_pred[mask])
        }


class ShortTermForecaster:
    """短期预测主类，支持模型切换"""

    def __init__(
        self,
        model_type: Literal['lightgbm', 'lstm'] = 'lightgbm',
        target_col: str = 'load',
        feature_cols: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        初始化短期预测器

        Args:
            model_type: 模型类型，'lightgbm'或'lstm'
            target_col: 目标列名
            feature_cols: 特征列名列表
            **kwargs: 模型特定参数
        """
        self.model_type = model_type
        self.target_col = target_col
        self.feature_cols = feature_cols

        if model_type == 'lightgbm':
            self.model = LightGBMForecaster(
                target_col=target_col,
                feature_cols=feature_cols,
                **kwargs
            )
        elif model_type == 'lstm':
            self.model = LSTMForecaster(
                target_col=target_col,
                feature_cols=feature_cols,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

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

        dt_features = datetime_features(pd.DatetimeIndex(df[datetime_col]))
        df = pd.concat([df, dt_features.reset_index(drop=True)], axis=1)

        if self.feature_cols is None:
            self.feature_cols = [
                col for col in df.columns
                if col != self.target_col and col != datetime_col
                and pd.api.types.is_numeric_dtype(df[col])
            ]

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        时间序列分割

        Args:
            df: 完整数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            训练集、验证集、测试集
        """
        return time_series_split(df, train_ratio, val_ratio)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> 'ShortTermForecaster':
        """
        训练模型

        Args:
            train_df: 训练数据
            val_df: 验证数据
            **kwargs: 训练参数

        Returns:
            训练后的实例
        """
        self.model.fit(train_df, val_df, **kwargs)
        return self

    def predict(self, df: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        预测

        Args:
            df: 输入数据

        Returns:
            预测结果
        """
        return self.model.predict(df)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型

        Args:
            test_df: 测试数据

        Returns:
            评估指标
        """
        y_true = test_df[self.target_col].values
        y_pred = self.predict(test_df)

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred['prediction'].values if 'prediction' in y_pred.columns else y_pred['q50'].values

        return self.model.calculate_metrics(y_true, y_pred)

    def switch_model(self, model_type: Literal['lightgbm', 'lstm'], **kwargs: Any) -> None:
        """
        切换模型

        Args:
            model_type: 新的模型类型
            **kwargs: 模型参数
        """
        self.model_type = model_type
        if model_type == 'lightgbm':
            self.model = LightGBMForecaster(
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                **kwargs
            )
        elif model_type == 'lstm':
            self.model = LSTMForecaster(
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
