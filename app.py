import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    read_csv_from_fileobj, load_load_data, load_calendar_data,
    standardize_column_names, validate_data
)
from src.data_quality import (
    calculate_missing_rate, detect_outliers, check_time_continuity,
    detect_duplicates, calculate_quality_score, generate_quality_report,
    clean_data
)
from src.feature_engineering import build_features
from src.short_term_forecast import ShortTermForecaster
from src.ultra_short_forecast import UltraShortTermForecaster
from src.peak_valley_analysis import PeakValleyAnalyzer
from src.demand_response import (
    DemandResponseOptimizer, DRResource, DRStrategy,
    create_default_resources, analyze_dr_potential
)
from src.error_analysis import ErrorAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.visualization import (
    plot_time_series, plot_prediction_comparison, plot_feature_importance,
    plot_error_distribution, plot_error_vs_temperature, plot_peak_valley,
    plot_dispatch_comparison, plot_resource_stack, plot_missing_heatmap,
    plot_model_comparison, plot_period_mape_comparison,
    st_plot
)
from src.utils import (
    mape, rmse, mae, format_number, format_percentage,
    calculate_all_metrics, calculate_period_mape, recommend_models
)

st.set_page_config(
    page_title="区域电网负荷预测与需求响应优化分析系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A5F;
    }
    h1, h2, h3 {
        color: #1E3A5F;
    }
    .stButton>button {
        background-color: #1E3A5F;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2c5282;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

if 'load_data' not in st.session_state:
    st.session_state.load_data = None
if 'calendar_data' not in st.session_state:
    st.session_state.calendar_data = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = None
if 'dr_result' not in st.session_state:
    st.session_state.dr_result = None
if 'error_analyzer' not in st.session_state:
    st.session_state.error_analyzer = None
if 'peak_valley_result' not in st.session_state:
    st.session_state.peak_valley_result = None
if 'ultra_short_result' not in st.session_state:
    st.session_state.ultra_short_result = None
if 'short_term_model' not in st.session_state:
    st.session_state.short_term_model = None
if 'model_comparison_result' not in st.session_state:
    st.session_state.model_comparison_result = None
if 'anomaly_detection_result' not in st.session_state:
    st.session_state.anomaly_detection_result = None

def page_data_import():
    st.header("📊 数据导入与质量检查")
    
    st.markdown("---")
    st.subheader("1. 数据上传")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 负荷气象数据")
        load_file = st.file_uploader(
            "上传负荷气象数据CSV文件",
            type=['csv'],
            key='load_uploader',
            help="包含字段：datetime, active_power_MW, reactive_power_MVar, voltage_kV, temperature_C, humidity_pct, solar_irradiance_Wm2, wind_speed_ms"
        )
        
        if load_file is not None:
            try:
                load_df = read_csv_from_fileobj(load_file, standardize_names=False)
                validation = validate_data(load_df, 'load')
                
                if not validation['is_valid']:
                    st.error(f"数据校验错误：{'; '.join(validation['errors'])}")
                else:
                    if validation['warnings']:
                        st.warning(f"数据校验警告：{'; '.join(validation['warnings'])}")
                    
                    load_df['datetime'] = pd.to_datetime(load_df['datetime'])
                    load_df = load_df.sort_values('datetime').reset_index(drop=True)
                    
                    st.session_state.load_data = load_df
                    st.success(f"✅ 成功加载负荷气象数据：{len(load_df)} 条记录")
                    st.dataframe(load_df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"文件读取失败：{str(e)}")
    
    with col2:
        st.markdown("#### 日历信息表")
        calendar_file = st.file_uploader(
            "上传日历信息表CSV文件",
            type=['csv'],
            key='calendar_uploader',
            help="包含字段：date, day_type, is_holiday, special_event"
        )
        
        if calendar_file is not None:
            try:
                cal_df = read_csv_from_fileobj(calendar_file, standardize_names=False)
                validation = validate_data(cal_df, 'calendar')
                
                if not validation['is_valid']:
                    st.error(f"数据校验错误：{'; '.join(validation['errors'])}")
                else:
                    if validation['warnings']:
                        st.warning(f"数据校验警告：{'; '.join(validation['warnings'])}")
                    
                    cal_df['date'] = pd.to_datetime(cal_df['date']).dt.date
                    
                    st.session_state.calendar_data = cal_df
                    st.success(f"✅ 成功加载日历信息：{len(cal_df)} 天")
                    st.dataframe(cal_df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"文件读取失败：{str(e)}")
    
    st.markdown("---")
    
    if st.session_state.load_data is not None:
        st.subheader("2. 数据质量检查")
        
        load_df = st.session_state.load_data
        
        load_df_for_check = load_df.reset_index() if load_df.index.name == 'datetime' else load_df
        
        quality_score = calculate_quality_score(load_df_for_check)
        missing_rate_df = calculate_missing_rate(load_df)
        outliers_info = detect_outliers(load_df)
        continuity_info = check_time_continuity(load_df_for_check)
        duplicates_info = detect_duplicates(load_df)
        
        quality_report = generate_quality_report(load_df_for_check)
        
        is_continuous = continuity_info.get('是否连续', False)
        gaps = continuity_info.get('间隔列表', [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score_color = "#28a745" if quality_score >= 80 else "#ffc107" if quality_score >= 60 else "#dc3545"
            st.metric("数据质量评分", f"{quality_score:.1f}/100")
            
        with col2:
            total_missing = int(missing_rate_df['缺失数量'].sum())
            missing_pct = (total_missing / (len(load_df) * len(load_df.columns))) * 100
            st.metric("缺失数据", f"{total_missing} ({missing_pct:.2f}%)")
            
        with col3:
            n_outliers = sum(v['异常值数量'] for v in outliers_info.values())
            outlier_pct = (n_outliers / len(load_df)) * 100
            st.metric("异常值数量", f"{n_outliers} ({outlier_pct:.2f}%)")
            
        with col4:
            continuity_status = "✅ 连续" if is_continuous else "❌ 不连续"
            st.metric("时间连续性", continuity_status)
            if not is_continuous:
                st.info(f"发现 {continuity_info.get('间隔数量', 0)} 个时间间隔")
        
        st.markdown("##### 质量检查报告")
        st.dataframe(quality_report, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 字段缺失率")
            fig = plot_missing_heatmap(missing_rate_df)
            st_plot(fig, use_container_width=True)
            
        with col2:
            st.markdown("##### 异常值分布")
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=load_df['active_power_MW'].values,
                name='有功负荷MW',
                boxpoints='outliers',
                marker_color='#1E3A5F'
            ))
            fig.update_layout(
                title='有功负荷异常值检测(IQR方法)',
                yaxis_title='有功负荷 MW'
            )
            st_plot(fig, use_container_width=True)
        
        if not is_continuous and gaps:
            st.warning(f"⚠️ 发现 {len(gaps)} 个时间间隔，前5个：")
            gaps_df = pd.DataFrame(gaps[:5], columns=['起始时间', '结束时间'])
            st.dataframe(gaps_df)
        
        if duplicates_info['重复数量'] > 0:
            st.warning(f"⚠️ 发现 {duplicates_info['重复数量']} 条重复数据")
        
        st.markdown("---")
        st.subheader("3. 数据清洗")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fill_method = st.selectbox(
                "缺失值填充方法",
                ['interpolate', 'mean', 'median', 'ffill', 'bfill'],
                index=0
            )
        with col2:
            outlier_method = st.selectbox(
                "异常值处理方法",
                ['clip', 'remove', 'nan'],
                index=0
            )
        with col3:
            remove_dup = st.checkbox("删除重复数据", value=True)
        
        if st.button("执行数据清洗"):
            with st.spinner("正在清洗数据..."):
                cleaned_df = clean_data(
                    load_df,
                    missing_strategy=fill_method,
                    outlier_method=outlier_method,
                    remove_dupes=remove_dup
                )
                
                st.session_state.load_data = cleaned_df
                new_score = calculate_quality_score(
                    cleaned_df.reset_index() if cleaned_df.index.name == 'datetime' else cleaned_df
                )
                
                st.success(f"✅ 数据清洗完成！数据质量评分：{quality_score:.1f} → {new_score:.1f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("清洗前数据量", len(load_df))
                with col2:
                    st.metric("清洗后数据量", len(cleaned_df))
        
        st.markdown("---")
        st.subheader("4. 时序概览")
        
        numeric_cols = load_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "选择要展示的字段",
            numeric_cols,
            default=['active_power_MW', 'temperature_C']
        )
        
        if selected_cols:
            fig = plot_time_series(
                load_df,
                'datetime',
                selected_cols,
                title='时序数据概览'
            )
            st_plot(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("5. 特征工程")
        
        if st.button("生成预测特征"):
            with st.spinner("正在生成特征..."):
                try:
                    feat_df = load_df.reset_index() if load_df.index.name == 'datetime' else load_df.copy()
                    features_df = build_features(
                        feat_df,
                        target_col='active_power_MW',
                        freq='15min'
                    )
                    
                    st.session_state.features_df = features_df
                    st.success(f"✅ 特征生成完成！共 {features_df.shape[1]} 个特征")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### 特征列表")
                        st.dataframe(
                            pd.DataFrame({
                                '特征名称': features_df.columns,
                                '特征类型': [
                                    '时间特征' if any(c in col for c in ['hour', 'day', 'month', 'weekend', 'sin', 'cos'])
                                    else '历史负荷特征' if 'lag' in col
                                    else '滑动均值特征' if 'rolling' in col
                                    else '气象特征' if any(w in col for w in ['temp', 'humidity', 'solar', 'wind'])
                                    else '日历特征' if any(d in col for d in ['holiday', 'day_type'])
                                    else '其他'
                                    for col in features_df.columns
                                ]
                            }),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("##### 特征相关性热力图")
                        corr = features_df.corr()
                        fig = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig.update_layout(
                            title='特征相关性矩阵',
                            height=600
                        )
                        st_plot(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"特征生成失败：{str(e)}")


def page_short_term_forecast():
    st.header("📈 短期负荷预测(日前)")
    
    if st.session_state.features_df is None or st.session_state.load_data is None:
        st.warning("⚠️ 请先在'数据导入'页面完成数据导入和特征生成")
        return
    
    st.markdown("---")
    st.subheader("1. 模型配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "选择预测模型",
            ['LightGBM', 'LSTM'],
            index=0
        )
        
        target_col = st.selectbox(
            "预测目标字段",
            ['active_power_MW'],
            index=0
        )
        
        train_ratio = st.slider(
            "训练集比例",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05
        )
        
    with col2:
        forecast_horizon = st.selectbox(
            "预测时段",
            ['24小时(逐小时)', '24小时(15分钟)'],
            index=0
        )
        
        input_days = st.selectbox(
            "历史输入天数",
            [7, 14, 21],
            index=1
        )
        
        if model_type == 'LightGBM':
            num_leaves = st.slider("num_leaves", 31, 127, 63)
            learning_rate = st.slider("learning_rate", 0.01, 0.1, 0.05, 0.01)
        else:
            lstm_units = st.slider("LSTM隐藏单元数", 32, 256, 128, 32)
            dropout_rate = st.slider("Dropout率", 0.1, 0.5, 0.2, 0.05)
    
    st.markdown("---")
    st.subheader("2. 模型训练")
    
    if st.button("🚀 开始训练模型", type="primary"):
        with st.spinner(f"正在训练{model_type}模型..."):
            try:
                features_df = st.session_state.features_df
                load_df = st.session_state.load_data
                
                if 'datetime' in features_df.columns:
                    features_df = features_df.set_index('datetime')
                
                load_indexed = load_df.set_index('datetime') if 'datetime' in load_df.columns else load_df
                
                aligned_dates = features_df.index.intersection(load_indexed.index)
                features_df = features_df.loc[aligned_dates]
                target = load_indexed.loc[aligned_dates, target_col]
                
                feature_cols = [col for col in features_df.columns if col != target_col]
                X = features_df[feature_cols].values
                y = target.values
                
                train_size = int(len(X) * train_ratio)
                val_size = int(len(X) * 0.1)
                test_size = len(X) - train_size - val_size
                
                X_train, y_train = X[:train_size], y[:train_size]
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
                test_dates = features_df.index[train_size+val_size:]
                
                if model_type == 'LightGBM':
                    from lightgbm import LGBMRegressor
                    
                    params = {
                        'num_leaves': num_leaves,
                        'learning_rate': learning_rate,
                        'n_estimators': 1000,
                        'objective': 'regression',
                        'metric': ['mape', 'rmse'],
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'verbose': -1
                    }
                    
                    model = LGBMRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[]
                    )
                    
                    y_pred = model.predict(X_test)
                    feature_importance = model.feature_importances_
                    
                    st.session_state.short_term_model = {
                        'model': model,
                        'model_type': 'LightGBM',
                        'feature_cols': feature_cols,
                        'feature_importance': feature_importance
                    }
                    
                else:
                    import torch
                    import torch.nn as nn
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    seq_length = 96
                    n_features = len(feature_cols)
                    
                    def create_sequences(X, y, seq_len):
                        X_seq, y_seq = [], []
                        for i in range(len(X) - seq_len):
                            X_seq.append(X[i:i+seq_len])
                            y_seq.append(y[i+seq_len])
                        return np.array(X_seq), np.array(y_seq)
                    
                    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
                    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
                    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
                    test_dates = test_dates[seq_length:]
                    
                    class LSTMModel(nn.Module):
                        def __init__(self, input_dim, hidden_dim, dropout_rate):
                            super().__init__()
                            self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
                            self.dropout1 = nn.Dropout(dropout_rate)
                            self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True, bidirectional=False)
                            self.dropout2 = nn.Dropout(dropout_rate)
                            self.fc1 = nn.Linear(hidden_dim//2, 32)
                            self.relu = nn.ReLU()
                            self.fc2 = nn.Linear(32, 1)
                            
                        def forward(self, x):
                            x, _ = self.lstm1(x)
                            x = self.dropout1(x)
                            x, _ = self.lstm2(x)
                            x = self.dropout2(x[:, -1, :])
                            x = self.fc1(x)
                            x = self.relu(x)
                            x = self.fc2(x)
                            return x.squeeze()
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = LSTMModel(n_features, lstm_units, dropout_rate).to(device)
                    
                    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
                    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
                    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
                    y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
                    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.MSELoss()
                    
                    batch_size = 64
                    dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    
                    epochs = 50
                    best_val_loss = float('inf')
                    patience = 10
                    counter = 0
                    
                    progress_bar = st.progress(0)
                    for epoch in range(epochs):
                        model.train()
                        for batch_X, batch_y in dataloader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                        
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_val_tensor)
                            val_loss = criterion(val_outputs, y_val_tensor)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                break
                        
                        progress_bar.progress((epoch + 1) / epochs)
                    
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(X_test_tensor).cpu().numpy()
                    y_test = y_test_seq
                    
                    st.session_state.short_term_model = {
                        'model': model,
                        'model_type': 'LSTM',
                        'feature_cols': feature_cols,
                        'seq_length': seq_length,
                        'device': device
                    }
                
                test_mape = mape(y_test, y_pred)
                test_rmse = rmse(y_test, y_pred)
                
                result_df = pd.DataFrame({
                    'datetime': test_dates,
                    'actual': y_test,
                    'predicted': y_pred
                })
                
                st.session_state.forecast_result = {
                    'result_df': result_df,
                    'mape': test_mape,
                    'rmse': test_rmse,
                    'model_type': model_type
                }
                
                if st.session_state.error_analyzer is None:
                    st.session_state.error_analyzer = ErrorAnalyzer()
                
                daily_errors = st.session_state.error_analyzer.calculate_daily_mape(result_df)
                
                st.success("✅ 模型训练完成！")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("测试集MAPE", f"{test_mape:.2f}%")
                with col2:
                    st.metric("测试集RMSE", f"{test_rmse:.2f} MW")
                with col3:
                    st.metric("预测样本数", len(y_test))
                
            except Exception as e:
                st.error(f"模型训练失败：{str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    st.markdown("---")
    
    if st.session_state.forecast_result is not None:
        st.subheader("3. 预测结果分析")
        
        result = st.session_state.forecast_result
        result_df = result['result_df']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### 预测值与实际值对比")
            fig = plot_prediction_comparison(
                result_df,
                'datetime',
                'actual',
                'predicted',
                title=f"{result['model_type']} 负荷预测结果"
            )
            st_plot(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 预测误差分析")
            result_df['error'] = result_df['predicted'] - result_df['actual']
            result_df['error_pct'] = np.abs(result_df['error'] / result_df['actual']) * 100
            
            fig = plot_error_distribution(result_df['error_pct'], title='MAPE分布')
            st_plot(fig, use_container_width=True)
        
        st.markdown("---")
        
        if result['model_type'] == 'LightGBM' and st.session_state.short_term_model is not None:
            st.subheader("4. 特征重要性")
            
            feature_cols = st.session_state.short_term_model['feature_cols']
            importance = st.session_state.short_term_model['feature_importance']
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False).head(20)
            
            fig = plot_feature_importance(importance_df, 'feature', 'importance')
            st_plot(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("5. 预测结果详情")
        
        display_days = st.slider("展示最近天数", 1, 7, 3)
        recent_df = result_df.tail(display_days * 96)
        
        st.dataframe(
            recent_df[['datetime', 'actual', 'predicted', 'error', 'error_pct']].style.format({
                'actual': '{:.2f}',
                'predicted': '{:.2f}',
                'error': '{:.2f}',
                'error_pct': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    st.markdown("---")
    st.subheader("6. 模型对比分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        compare_train_ratio = st.slider(
            "训练集比例(对比)",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05,
            key='compare_train_ratio'
        )
    with col2:
        compare_days = st.slider(
            "对比测试天数",
            min_value=1,
            max_value=14,
            value=7,
            key='compare_days'
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 开始模型对比", type="primary", key='compare_button'):
            with st.spinner("正在训练双模型并进行对比分析..."):
                try:
                    features_df = st.session_state.features_df
                    load_df = st.session_state.load_data
                    target_col = 'active_power_MW'
                    
                    if 'datetime' in features_df.columns:
                        features_df = features_df.set_index('datetime')
                    
                    load_indexed = load_df.set_index('datetime') if 'datetime' in load_df.columns else load_df
                    
                    aligned_dates = features_df.index.intersection(load_indexed.index)
                    features_df = features_df.loc[aligned_dates]
                    target = load_indexed.loc[aligned_dates, target_col]
                    
                    feature_cols = [col for col in features_df.columns if col != target_col]
                    X = features_df[feature_cols].values
                    y = target.values
                    
                    test_samples = min(compare_days * 96, len(X) - int(len(X) * compare_train_ratio))
                    train_size = len(X) - test_samples
                    
                    X_train, y_train = X[:train_size], y[:train_size]
                    X_test, y_test = X[train_size:], y[train_size:]
                    test_dates = features_df.index[train_size:]
                    
                    lgbm_params = {
                        'num_leaves': 63,
                        'learning_rate': 0.05,
                        'n_estimators': 1000,
                        'objective': 'regression',
                        'metric': ['mape', 'rmse'],
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'verbose': -1
                    }
                    
                    from lightgbm import LGBMRegressor
                    lgbm_model = LGBMRegressor(**lgbm_params)
                    lgbm_model.fit(X_train, y_train, callbacks=[])
                    lgbm_pred = lgbm_model.predict(X_test)
                    
                    seq_length = 96
                    n_features = len(feature_cols)
                    
                    def create_sequences(X, y, seq_len):
                        X_seq, y_seq = [], []
                        for i in range(len(X) - seq_len):
                            X_seq.append(X[i:i+seq_len])
                            y_seq.append(y[i+seq_len])
                        return np.array(X_seq), np.array(y_seq)
                    
                    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
                    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
                    test_dates_lstm = test_dates[seq_length:]
                    
                    import torch
                    import torch.nn as nn
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    class LSTMModel(nn.Module):
                        def __init__(self, input_dim, hidden_dim, dropout_rate):
                            super().__init__()
                            self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
                            self.dropout1 = nn.Dropout(dropout_rate)
                            self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True, bidirectional=False)
                            self.dropout2 = nn.Dropout(dropout_rate)
                            self.fc1 = nn.Linear(hidden_dim//2, 32)
                            self.relu = nn.ReLU()
                            self.fc2 = nn.Linear(32, 1)
                            
                        def forward(self, x):
                            x, _ = self.lstm1(x)
                            x = self.dropout1(x)
                            x, _ = self.lstm2(x)
                            x = self.dropout2(x[:, -1, :])
                            x = self.fc1(x)
                            x = self.relu(x)
                            x = self.fc2(x)
                            return x.squeeze()
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    lstm_model = LSTMModel(n_features, 128, 0.2).to(device)
                    
                    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
                    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
                    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
                    
                    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                    criterion = nn.MSELoss()
                    
                    batch_size = 64
                    dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    
                    epochs = 50
                    best_val_loss = float('inf')
                    patience = 10
                    counter = 0
                    
                    val_size = int(len(X_train_seq) * 0.1)
                    X_val_tensor = X_train_tensor[-val_size:]
                    y_val_tensor = y_train_tensor[-val_size:]
                    X_train_tensor = X_train_tensor[:-val_size]
                    y_train_tensor = y_train_tensor[:-val_size]
                    
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    
                    for epoch in range(epochs):
                        lstm_model.train()
                        for batch_X, batch_y in train_dataloader:
                            optimizer.zero_grad()
                            outputs = lstm_model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                        
                        lstm_model.eval()
                        with torch.no_grad():
                            val_outputs = lstm_model(X_val_tensor)
                            val_loss = criterion(val_outputs, y_val_tensor)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                break
                    
                    lstm_model.eval()
                    with torch.no_grad():
                        lstm_pred = lstm_model(X_test_tensor).cpu().numpy()
                    
                    common_len = min(len(lgbm_pred) - seq_length, len(lstm_pred))
                    lgbm_pred_aligned = lgbm_pred[seq_length:seq_length + common_len]
                    y_test_aligned = y_test[seq_length:seq_length + common_len]
                    test_dates_aligned = test_dates[seq_length:seq_length + common_len]
                    
                    lgbm_metrics = calculate_all_metrics(y_test_aligned, lgbm_pred_aligned)
                    lstm_metrics = calculate_all_metrics(y_test_aligned, lstm_pred)
                    
                    comparison_df = pd.DataFrame({
                        'datetime': test_dates_aligned,
                        'actual': y_test_aligned,
                        'lightgbm_pred': lgbm_pred_aligned,
                        'lstm_pred': lstm_pred
                    })
                    
                    lgbm_period_mapes = calculate_period_mape(
                        comparison_df, 'datetime', 'actual', 'lightgbm_pred'
                    )
                    lstm_period_mapes = calculate_period_mape(
                        comparison_df, 'datetime', 'actual', 'lstm_pred'
                    )
                    
                    recommendations, recommendation_summary = recommend_models(
                        lgbm_period_mapes, lstm_period_mapes
                    )
                    
                    st.session_state.model_comparison_result = {
                        'comparison_df': comparison_df,
                        'lgbm_metrics': lgbm_metrics,
                        'lstm_metrics': lstm_metrics,
                        'lgbm_period_mapes': lgbm_period_mapes,
                        'lstm_period_mapes': lstm_period_mapes,
                        'recommendations': recommendations,
                        'recommendation_summary': recommendation_summary
                    }
                    
                    st.success("✅ 模型对比分析完成！")
                    
                except Exception as e:
                    st.error(f"模型对比失败：{str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    if st.session_state.model_comparison_result is not None:
        comp_result = st.session_state.model_comparison_result
        comparison_df = comp_result['comparison_df']
        lgbm_metrics = comp_result['lgbm_metrics']
        lstm_metrics = comp_result['lstm_metrics']
        lgbm_period_mapes = comp_result['lgbm_period_mapes']
        lstm_period_mapes = comp_result['lstm_period_mapes']
        recommendations = comp_result['recommendations']
        recommendation_summary = comp_result['recommendation_summary']
        
        st.markdown("##### 6.1 预测曲线对比")
        fig = plot_model_comparison(
            comparison_df,
            'datetime',
            'actual',
            'lightgbm_pred',
            'lstm_pred',
            lgbm_metrics['mape'],
            lstm_metrics['mape'],
            title='LightGBM vs LSTM 预测结果对比'
        )
        st_plot(fig, use_container_width=True)
        
        st.markdown("##### 6.2 综合指标对比")
        
        metrics_table = pd.DataFrame({
            '指标': ['MAPE (%)', 'RMSE (MW)', 'MAE (MW)', '最大单点误差 (%)', '预测偏移方向'],
            'LightGBM': [
                f"{lgbm_metrics['mape']:.2f}%",
                f"{lgbm_metrics['rmse']:.2f}",
                f"{lgbm_metrics['mae']:.2f}",
                f"{lgbm_metrics['max_ape']:.2f}%",
                lgbm_metrics['bias']
            ],
            'LSTM': [
                f"{lstm_metrics['mape']:.2f}%",
                f"{lstm_metrics['rmse']:.2f}",
                f"{lstm_metrics['mae']:.2f}",
                f"{lstm_metrics['max_ape']:.2f}%",
                lstm_metrics['bias']
            ]
        })
        
        def highlight_recommended_row(row):
            if row.name == 'MAPE (%)':
                lgbm_val = lgbm_metrics['mape']
                lstm_val = lstm_metrics['mape']
                if lgbm_val < lstm_val:
                    return ['', 'background-color: #d4edda; color: #155724', '']
                else:
                    return ['', '', 'background-color: #d4edda; color: #155724']
            elif row.name == 'RMSE (MW)':
                lgbm_val = lgbm_metrics['rmse']
                lstm_val = lstm_metrics['rmse']
                if lgbm_val < lstm_val:
                    return ['', 'background-color: #d4edda; color: #155724', '']
                else:
                    return ['', '', 'background-color: #d4edda; color: #155724']
            elif row.name == 'MAE (MW)':
                lgbm_val = lgbm_metrics['mae']
                lstm_val = lstm_metrics['mae']
                if lgbm_val < lstm_val:
                    return ['', 'background-color: #d4edda; color: #155724', '']
                else:
                    return ['', '', 'background-color: #d4edda; color: #155724']
            elif row.name == '最大单点误差 (%)':
                lgbm_val = lgbm_metrics['max_ape']
                lstm_val = lstm_metrics['max_ape']
                if lgbm_val < lstm_val:
                    return ['', 'background-color: #d4edda; color: #155724', '']
                else:
                    return ['', '', 'background-color: #d4edda; color: #155724']
            return ['', '', '']
        
        metrics_table_styled = metrics_table.style.apply(highlight_recommended_row, axis=1)
        st.table(metrics_table_styled)
        
        st.markdown("##### 6.3 分时段误差对比")
        fig = plot_period_mape_comparison(
            lgbm_period_mapes,
            lstm_period_mapes,
            title='各时段MAPE对比'
        )
        st_plot(fig, use_container_width=True)
        
        st.markdown("##### 6.4 分时段模型推荐")
        
        periods = ["谷时(0-7点)", "平时(7-11,15-19点)", "峰时(11-15,19-22点)", "深谷(22-24点)"]
        recommendation_data = []
        for period in periods:
            recommendation_data.append({
                '时段': period,
                'LightGBM MAPE': f"{lgbm_period_mapes[period]:.2f}%",
                'LSTM MAPE': f"{lstm_period_mapes[period]:.2f}%",
                '推荐模型': recommendations[period]
            })
        
        rec_table = pd.DataFrame(recommendation_data)
        
        def highlight_recommended_model(row):
            styles = [''] * len(row)
            if row['推荐模型'] == 'LightGBM':
                styles[1] = 'background-color: #d4edda; color: #155724'
            else:
                styles[2] = 'background-color: #d4edda; color: #155724'
            styles[3] = 'background-color: #d4edda; color: #155724; font-weight: bold'
            return styles
        
        rec_table_styled = rec_table.style.apply(highlight_recommended_model, axis=1)
        st.table(rec_table_styled)
        
        st.markdown("##### 6.5 推荐策略总结")
        st.info(f"📊 {recommendation_summary}")


def page_ultra_short_forecast():
    st.header("⚡ 超短期预测(4小时)")
    
    if st.session_state.load_data is None:
        st.warning("⚠️ 请先在'数据导入'页面完成数据导入")
        return
    
    st.markdown("---")
    st.subheader("1. 预测配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox(
            "预测目标",
            ['active_power_MW'],
            key='ultra_target'
        )
        
        input_window = st.selectbox(
            "输入窗口长度",
            ['1小时(4点)', '2小时(8点)'],
            index=1
        )
        
    with col2:
        forecast_horizon = st.selectbox(
            "预测时段",
            ['1小时(4点)', '2小时(8点)', '4小时(16点)'],
            index=2
        )
        
        update_interval = st.selectbox(
            "在线更新间隔",
            ['15分钟', '1小时'],
            index=0
        )
    
    st.markdown("---")
    st.subheader("2. 超短期滚动预测")
    
    load_df = st.session_state.load_data
    target_data = load_df[target_col].copy()
    
    if st.button("🔄 执行超短期预测", type="primary"):
        with st.spinner("正在执行超短期预测..."):
            try:
                input_steps = 8 if input_window == '2小时(8点)' else 4
                forecast_steps = 16 if forecast_horizon == '4小时(16点)' else (8 if forecast_horizon == '2小时(8点)' else 4)
                
                last_idx = len(target_data) - forecast_steps - input_steps
                
                if last_idx < 0:
                    st.error("⚠️ 数据量不足，请确保有足够的历史数据")
                    return
                
                X = []
                y = []
                
                for i in range(input_steps, len(target_data) - forecast_steps + 1):
                    X.append(target_data.iloc[i-input_steps:i].values)
                    y.append(target_data.iloc[i:i+forecast_steps].values)
                
                X = np.array(X)
                y = np.array(y)
                
                train_size = int(len(X) * 0.8)
                X_train, y_train = X[:train_size], y[:train_size]
                X_test, y_test = X[train_size:], y[train_size:]
                
                from lightgbm import LGBMRegressor
                
                models = []
                for step in range(forecast_steps):
                    model = LGBMRegressor(
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=200,
                        verbose=-1
                    )
                    model.fit(X_train, y_train[:, step])
                    models.append(model)
                
                last_input = target_data.iloc[-input_steps:].values.reshape(1, -1)
                predictions = []
                for model in models:
                    pred = model.predict(last_input)[0]
                    predictions.append(pred)
                predictions = np.array(predictions)
                
                actual = target_data.iloc[-forecast_steps:].values
                future_dates = pd.date_range(
                    start=target_data.index[-forecast_steps],
                    periods=forecast_steps,
                    freq='15T'
                )
                
                lower_bound = predictions * 0.8
                upper_bound = predictions * 1.2
                
                result_df = pd.DataFrame({
                    'datetime': future_dates,
                    'actual': actual,
                    'predicted': predictions,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
                
                test_predictions = []
                for model in models:
                    test_predictions.append(model.predict(X_test))
                test_predictions = np.array(test_predictions).T
                
                test_mape = np.mean(np.abs((y_test - test_predictions) / (y_test + 1e-6))) * 100
                test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))
                
                st.session_state.ultra_short_result = {
                    'result_df': result_df,
                    'models': models,
                    'input_steps': input_steps,
                    'forecast_steps': forecast_steps,
                    'mape': test_mape,
                    'rmse': test_rmse,
                    'last_update': datetime.now()
                }
                
                st.success("✅ 超短期预测完成！")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("测试集MAPE", f"{test_mape:.2f}%")
                with col2:
                    st.metric("测试集RMSE", f"{test_rmse:.2f} MW")
                
            except Exception as e:
                st.error(f"预测失败：{str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    st.markdown("---")
    
    if st.session_state.ultra_short_result is not None:
        st.subheader("3. 预测结果")
        
        result = st.session_state.ultra_short_result
        result_df = result['result_df']
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=result_df['datetime'],
                y=result_df['actual'],
                mode='lines+markers',
                name='实际值',
                line=dict(color='#1E3A5F', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=result_df['datetime'],
                y=result_df['predicted'],
                mode='lines+markers',
                name='预测值',
                line=dict(color='#38B2AC', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=result_df['datetime'],
                y=result_df['upper_bound'],
                mode='lines',
                line=dict(color='rgba(56, 178, 172, 0.3)', width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=result_df['datetime'],
                y=result_df['lower_bound'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(56, 178, 172, 0.2)',
                line=dict(color='rgba(56, 178, 172, 0.3)', width=0),
                name='80%置信区间'
            ))
            
            fig.update_layout(
                title='超短期负荷预测结果',
                xaxis_title='时间',
                yaxis_title='有功负荷 MW',
                hovermode='x unified',
                height=500
            )
            
            st_plot(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 预测详情")
            result_df['误差%'] = np.abs((result_df['predicted'] - result_df['actual']) / result_df['actual']) * 100
            
            st.dataframe(
                result_df[['datetime', 'actual', 'predicted', '误差%']].style.format({
                    'actual': '{:.2f}',
                    'predicted': '{:.2f}',
                    '误差%': '{:.2f}%'
                }),
                use_container_width=True,
                height=450
            )
        
        st.markdown("---")
        st.subheader("4. 在线更新")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📅 上次模型更新时间：{result['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            new_data_points = st.number_input(
                "新增数据点数(15分钟/点)",
                min_value=1,
                max_value=50,
                value=4
            )
            
        with col2:
            if st.button("🔄 增量更新模型"):
                with st.spinner("正在增量更新模型..."):
                    try:
                        models = result['models']
                        input_steps = result['input_steps']
                        
                        if len(target_data) > input_steps + new_data_points:
                            new_X = []
                            new_y = []
                            
                            start_idx = len(target_data) - input_steps - new_data_points
                            for i in range(start_idx, len(target_data) - input_steps):
                                new_X.append(target_data.iloc[i:i+input_steps].values)
                                new_y.append(target_data.iloc[i+input_steps])
                            
                            new_X = np.array(new_X)
                            new_y = np.array(new_y)
                            
                            for i, model in enumerate(models):
                                if i < len(new_y):
                                    model.fit(new_X, new_y, init_model=model)
                            
                            result['models'] = models
                            result['last_update'] = datetime.now()
                            st.session_state.ultra_short_result = result
                            
                            st.success("✅ 模型增量更新完成！")
                        else:
                            st.warning("⚠️ 数据量不足，无法进行增量更新")
                    
                    except Exception as e:
                        st.error(f"更新失败：{str(e)}")


def page_peak_valley_analysis():
    st.header("📊 峰谷分析")
    
    if st.session_state.load_data is None:
        st.warning("⚠️ 请先在'数据导入'页面完成数据导入")
        return
    
    st.markdown("---")
    
    load_df = st.session_state.load_data
    target_col = 'active_power_MW'
    
    if st.button("🔍 执行峰谷分析", type="primary") or st.session_state.peak_valley_result is not None:
        
        if st.button("🔍 重新执行峰谷分析") or st.session_state.peak_valley_result is None:
            with st.spinner("正在执行峰谷分析..."):
                try:
                    analyzer = PeakValleyAnalyzer(target_col='active_power_MW')
                    
                    daily_features = analyzer.extract_daily_peak_valley(load_df)
                    
                    peak_duration = analyzer.calculate_peak_duration(load_df)
                    
                    peak_valley_result = {
                        'daily_features': daily_features,
                        'peak_duration': peak_duration,
                        'monthly_trend': analyzer.aggregate_monthly_features(daily_features)
                    }
                    
                    st.session_state.peak_valley_result = peak_valley_result
                    st.success("✅ 峰谷分析完成！")
                    
                except Exception as e:
                    st.error(f"分析失败：{str(e)}")
                    return
        
        result = st.session_state.peak_valley_result
        daily_features = result['daily_features']
        
        st.subheader("1. 日峰谷特征")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_peak = daily_features['peak_value'].mean()
            st.metric("平均峰值", f"{avg_peak:.2f} MW")
        
        with col2:
            avg_valley = daily_features['valley_value'].mean()
            st.metric("平均谷值", f"{avg_valley:.2f} MW")
        
        with col3:
            avg_diff = daily_features['peak_valley_diff'].mean()
            st.metric("平均峰谷差", f"{avg_diff:.2f} MW")
        
        with col4:
            avg_load_rate = daily_features['daily_load_rate'].mean() * 100
            st.metric("平均日负荷率", f"{avg_load_rate:.2f}%")
        
        st.markdown("##### 日峰谷特征详情")
        st.dataframe(
            daily_features.style.format({
                'peak_value': '{:.2f}',
                'valley_value': '{:.2f}',
                'peak_valley_diff': '{:.2f}',
                'mean_value': '{:.2f}',
                'daily_load_rate': '{:.2%}',
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("2. 典型日负荷曲线")
        
        date_options = daily_features['date'].tolist()
        selected_date = st.selectbox(
            "选择日期查看负荷曲线",
            date_options,
            index=len(date_options) - 1
        )
        
        selected_datetime = pd.to_datetime(selected_date)
        day_data = load_df[
            pd.to_datetime(load_df['datetime']).dt.date == selected_datetime.date()
        ].copy()
        
        if len(day_data) > 0:
            day_features = daily_features[daily_features['date'] == selected_date].iloc[0]
            
            fig = plot_peak_valley(
                day_data,
                'datetime',
                target_col,
                peak_time=day_features['peak_time'],
                valley_time=day_features['valley_time'],
                threshold=day_features['mean_value'] * 1.2,
                title=f"{selected_date} 日负荷曲线"
            )
            st_plot(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"🕐 峰值时刻：{day_features['peak_time']}")
            with col2:
                st.info(f"🕐 谷值时刻：{day_features['valley_time']}")
            with col3:
                st.info(f"📊 峰谷差：{day_features['peak_valley_diff']:.2f} MW")
            with col4:
                spike_h = result['peak_duration']['peak_duration_hours'].mean() if 'peak_duration' in result else 0
                st.info(f"⏱️ 平均尖峰时长：{spike_h:.2f} 小时")
        
        st.markdown("---")
        st.subheader("3. 月度趋势分析")
        
        monthly_trend = result['monthly_trend']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 月度峰谷值趋势")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_trend['month'],
                y=monthly_trend['avg_peak_value'],
                mode='lines+markers',
                name='平均峰值',
                line=dict(color='#dc3545', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_trend['month'],
                y=monthly_trend['avg_valley_value'],
                mode='lines+markers',
                name='平均谷值',
                line=dict(color='#28a745', width=2)
            ))
            
            fig.update_layout(
                title='月度峰谷值变化趋势',
                xaxis_title='月份',
                yaxis_title='有功负荷 MW',
                hovermode='x unified'
            )
            st_plot(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 月度峰谷差与负荷率趋势")
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=monthly_trend['month'],
                y=monthly_trend['avg_peak_valley_diff'],
                name='平均峰谷差',
                yaxis='y',
                marker_color='#38B2AC'
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_trend['month'],
                y=monthly_trend['avg_daily_load_rate'] * 100,
                mode='lines+markers',
                name='平均负荷率',
                yaxis='y2',
                line=dict(color='#ED8936', width=2)
            ))
            
            fig.update_layout(
                title='月度峰谷差与负荷率',
                xaxis_title='月份',
                yaxis=dict(title='峰谷差 MW'),
                yaxis2=dict(title='负荷率 %', overlaying='y', side='right'),
                hovermode='x unified',
                barmode='group'
            )
            st_plot(fig, use_container_width=True)
        
        st.dataframe(
            monthly_trend.style.format({
                'avg_peak_value_MW': '{:.2f}',
                'avg_valley_value_MW': '{:.2f}',
                'avg_mean_value_MW': '{:.2f}',
                'avg_peak_valley_diff_MW': '{:.2f}',
                'avg_load_rate': '{:.2%}',
                'avg_spike_duration_hours': '{:.2f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("4. 尖峰时段分析")
        
        peak_duration = result['peak_duration']
        
        if peak_duration is not None and len(peak_duration) > 0:
            avg_spike = peak_duration['peak_duration_hours'].mean()
            st.info(f"📊 共分析 {len(peak_duration)} 天，平均尖峰时长 {avg_spike:.2f} 小时（超过日均值120%）")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Histogram(
                    x=peak_duration['peak_duration_hours'],
                    nbinsx=20,
                    marker_color='#dc3545'
                )])
                fig.update_layout(
                    title='尖峰持续时间分布',
                    xaxis_title='持续时间(小时)',
                    yaxis_title='天数'
                )
                st_plot(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(
                    x=peak_duration['date'].astype(str),
                    y=peak_duration['peak_duration_hours'],
                    marker_color='#ED8936'
                )])
                fig.update_layout(
                    title='每日尖峰时长',
                    xaxis_title='日期',
                    yaxis_title='尖峰时长(小时)'
                )
                st_plot(fig, use_container_width=True)
            
            st.markdown("##### 尖峰时段详情（前10天）")
            st.dataframe(peak_duration.head(10), use_container_width=True)
        else:
            st.info("未识别到尖峰时段")


def page_demand_response():
    st.header("🔧 需求响应模拟")
    
    if st.session_state.load_data is None:
        st.warning("⚠️ 请先在'数据导入'页面完成数据导入")
        return
    
    st.markdown("---")
    
    load_df = st.session_state.load_data
    target_col = 'active_power_MW'
    
    if st.session_state.peak_valley_result is None:
        with st.spinner("正在分析负荷特性..."):
            dr_potential = analyze_dr_potential(load_df, target_col)
    else:
        daily_features = st.session_state.peak_valley_result['daily_features']
        dr_potential = {
            'peak_load_MW': daily_features['peak_value'].max(),
            'valley_load_MW': daily_features['valley_value'].min(),
            'mean_load_MW': daily_features['mean_value'].mean(),
            'peak_valley_diff_MW': daily_features['peak_valley_diff'].mean(),
            'shaving_potential_pct': (daily_features['peak_valley_diff'].mean() / daily_features['peak_value'].max() * 100)
        }
    
    st.subheader("1. 需求响应潜力分析")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("峰值负荷", f"{dr_potential['peak_load_MW']:.2f} MW")
    with col2:
        st.metric("谷值负荷", f"{dr_potential['valley_load_MW']:.2f} MW")
    with col3:
        st.metric("峰谷差", f"{dr_potential['peak_valley_diff_MW']:.2f} MW")
    with col4:
        st.metric("削峰潜力", f"{dr_potential['shaving_potential_pct']:.2f}%")
    
    st.markdown("---")
    st.subheader("2. 可削减负荷资源池配置")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏠 空调负荷")
        ac_share = st.slider("空调负荷占比", 0.1, 0.5, 0.3, 0.05)
        ac_capacity = st.slider("可调容量 MW", 10.0, 100.0, 50.0, 5.0)
        ac_max_duration = st.slider("最大持续时长 小时", 1.0, 8.0, 4.0, 0.5)
    
    with col2:
        st.markdown("#### 🏭 工业可中断负荷")
        ind_capacity = st.slider("可调容量 MW", 10.0, 80.0, 30.0, 5.0)
        ind_max_duration = st.slider("最大持续时长 小时", 0.5, 4.0, 2.0, 0.5)
        ind_cost = st.slider("补偿成本 元/MWh", 50.0, 200.0, 80.0, 10.0)
    
    with col3:
        st.markdown("#### 🔋 分布式储能")
        storage_power = st.slider("充放电功率 MW", 5.0, 50.0, 20.0, 5.0)
        storage_capacity = st.slider("储能容量 MWh", 20.0, 200.0, 100.0, 10.0)
        storage_efficiency = st.slider("充放电效率", 0.85, 0.95, 0.9, 0.01)
    
    st.markdown("---")
    st.subheader("3. 响应策略参数")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        peak_target = st.slider(
            "削峰目标值 MW",
            float(dr_potential['valley_load_MW']),
            float(dr_potential['peak_load_MW']),
            float(dr_potential['peak_load_MW'] * 0.9),
            1.0
        )
    
    with col2:
        response_start = st.slider("响应开始时刻", 0, 23, 10)
        response_end = st.slider("响应结束时刻", 0, 23, 22)
    
    with col3:
        max_curtail_duration = st.slider("单次最大削减时长 小时", 1.0, 6.0, 4.0, 0.5)
    
    with col4:
        max_temp_offset = st.slider("用户舒适度约束 温度偏移 ℃", 1.0, 5.0, 2.0, 0.5)
    
    st.markdown("##### 电价参数")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_price = st.number_input("尖峰电价 元/kWh", 0.5, 3.0, 1.2, 0.1)
    with col2:
        flat_price = st.number_input("平段电价 元/kWh", 0.3, 1.5, 0.7, 0.1)
    with col3:
        valley_price = st.number_input("谷段电价 元/kWh", 0.1, 0.8, 0.4, 0.1)
    
    st.markdown("---")
    st.subheader("4. 模拟调度日期")
    
    date_options = load_df.index.date
    date_options = sorted(list(set(date_options)))
    
    selected_dates = st.multiselect(
        "选择模拟调度的日期（可多选）",
        date_options,
        default=[date_options[-1]]
    )
    
    st.markdown("---")
    
    if st.button("🚀 运行需求响应模拟", type="primary") and len(selected_dates) > 0:
        with st.spinner("正在求解最优调度方案..."):
            try:
                optimizer = DemandResponseOptimizer()
                
                optimizer.add_resource(DRResource(
                    name='空调负荷',
                    resource_type='air_conditioning',
                    capacity_mw=ac_capacity,
                    max_duration_hours=ac_max_duration,
                    cost_per_mwh=50.0,
                    ac_share=ac_share
                ))
                
                optimizer.add_resource(DRResource(
                    name='工业可中断负荷',
                    resource_type='industrial',
                    capacity_mw=ind_capacity,
                    max_duration_hours=ind_max_duration,
                    cost_per_mwh=ind_cost,
                    ac_share=0.0
                ))
                
                optimizer.add_resource(DRResource(
                    name='分布式储能',
                    resource_type='storage',
                    capacity_mw=storage_power,
                    max_duration_hours=storage_capacity / storage_power if storage_power > 0 else 4,
                    cost_per_mwh=30.0,
                    ac_share=0.0
                ))
                
                strategy = DRStrategy(
                    peak_target_mw=peak_target,
                    response_start_hour=response_start,
                    response_end_hour=response_end,
                    max_curtailment_duration=max_curtail_duration,
                    max_temp_offset=max_temp_offset,
                    total_energy_conservation=True
                )
                optimizer.set_strategy(strategy)
                
                all_results = []
                
                for selected_date in selected_dates:
                    day_data = load_df[
                        (load_df.index.date == selected_date)
                    ].copy()
                    
                    if len(day_data) < 90:
                        st.warning(f"⚠️ {selected_date} 数据不足，跳过")
                        continue
                    
                    optimizer.time_index = day_data.index
                    
                    result = optimizer.optimize(
                        day_data,
                        load_col=target_col,
                        peak_price=peak_price,
                        valley_price=valley_price
                    )
                    
                    if result.success:
                        all_results.append({
                            'date': selected_date,
                            'result': result
                        })
                        st.success(f"✅ {selected_date} 优化求解成功！")
                    else:
                        st.warning(f"⚠️ {selected_date} 求解失败：{result.message}")
                
                if all_results:
                    st.session_state.dr_result = all_results
                    
                    if len(all_results) == 1:
                        result = all_results[0]['result']
                        
                        st.markdown("---")
                        st.subheader("5. 调度结果分析")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("削峰量", f"{result.peak_reduction_kw:.0f} kW")
                        with col2:
                            st.metric("填谷量", f"{result.valley_increase_kw:.0f} kW")
                        with col3:
                            st.metric("峰谷差(优化前)", f"{result.peak_valley_diff_before:.2f} MW")
                        with col4:
                            st.metric("峰谷差(优化后)", f"{result.peak_valley_diff_after:.2f} MW")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            improvement = (result.peak_valley_diff_before - result.peak_valley_diff_after) / result.peak_valley_diff_before * 100
                            st.metric("峰谷差改善", f"{improvement:.2f}%")
                        with col2:
                            st.metric("综合成本节省", f"{result.cost_savings:.2f} 元")
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("##### 调度前后负荷曲线对比")
                            fig = plot_dispatch_comparison(
                                result.scheduling_df,
                                'datetime',
                                'base_load_MW',
                                'optimized_load_MW',
                                title='调度前后负荷曲线对比'
                            )
                            st_plot(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("##### 各资源出力情况")
                            fig = plot_resource_stack(
                                result.scheduling_df,
                                'datetime',
                                ['ac_curtailment_MW', 'industrial_curtailment_MW', 'storage_discharge_MW'],
                                title='各资源出力时序'
                            )
                            st_plot(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("6. 储能充放电曲线")
                        
                        storage_df = result.scheduling_df[['datetime', 'storage_charge_MW', 'storage_discharge_MW']].copy()
                        storage_df['net_power_MW'] = storage_df['storage_discharge_MW'] - storage_df['storage_charge_MW']
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=storage_df['datetime'],
                            y=storage_df['storage_charge_MW'],
                            name='充电',
                            marker_color='#28a745'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=storage_df['datetime'],
                            y=-storage_df['storage_discharge_MW'],
                            name='放电',
                            marker_color='#dc3545'
                        ))
                        
                        fig.add_hline(y=0, line=dict(color='black', width=1))
                        
                        fig.update_layout(
                            title='储能充放电功率',
                            xaxis_title='时间',
                            yaxis_title='功率 MW',
                            barmode='relative',
                            bargap=0
                        )
                        st_plot(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("7. 调度详情")
                        
                        st.dataframe(
                            result.scheduling_df.style.format({
                                'base_load_MW': '{:.2f}',
                                'optimized_load_MW': '{:.2f}',
                                'ac_curtailment_MW': '{:.2f}',
                                'industrial_curtailment_MW': '{:.2f}',
                                'storage_charge_MW': '{:.2f}',
                                'storage_discharge_MW': '{:.2f}',
                                'load_delta_MW': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.markdown("---")
                        st.subheader("5. 多日调度结果汇总")
                        
                        summary_data = []
                        for item in all_results:
                            r = item['result']
                            summary_data.append({
                                '日期': item['date'],
                                '削峰量_kW': r.peak_reduction_kw,
                                '填谷量_kW': r.valley_increase_kw,
                                '峰谷差_前_MW': r.peak_valley_diff_before,
                                '峰谷差_后_MW': r.peak_valley_diff_after,
                                '改善率_%': (r.peak_valley_diff_before - r.peak_valley_diff_after) / r.peak_valley_diff_before * 100,
                                '成本节省_元': r.cost_savings
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(
                            summary_df.style.format({
                                '削峰量_kW': '{:.0f}',
                                '填谷量_kW': '{:.0f}',
                                '峰谷差_前_MW': '{:.2f}',
                                '峰谷差_后_MW': '{:.2f}',
                                '改善率_%': '{:.2f}%',
                                '成本节省_元': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_improvement = summary_df['改善率_%'].mean()
                            st.metric("平均峰谷差改善", f"{avg_improvement:.2f}%")
                        with col2:
                            total_savings = summary_df['成本节省_元'].sum()
                            st.metric("累计成本节省", f"{total_savings:.2f} 元")
                        with col3:
                            avg_peak_shaving = summary_df['削峰量_kW'].mean()
                            st.metric("平均削峰量", f"{avg_peak_shaving:.0f} kW")
                    
            except Exception as e:
                st.error(f"模拟失败：{str(e)}")
                import traceback
                st.error(traceback.format_exc())


def page_error_analysis():
    st.header("📉 预测误差分析")
    
    if st.session_state.forecast_result is None:
        st.warning("⚠️ 请先在'短期预测'页面完成模型训练")
        return
    
    st.markdown("---")
    
    result = st.session_state.forecast_result
    result_df = result['result_df'].copy()
    
    if st.session_state.error_analyzer is None:
        st.session_state.error_analyzer = ErrorAnalyzer()
    
    analyzer = st.session_state.error_analyzer
    
    st.subheader("1. 日MAPE统计")
    
    daily_mape = analyzer.calculate_daily_mape(result_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_mape = daily_mape['mape'].mean()
        st.metric("平均MAPE", f"{avg_mape:.2f}%")
    with col2:
        max_mape = daily_mape['mape'].max()
        st.metric("最大MAPE", f"{max_mape:.2f}%")
    with col3:
        min_mape = daily_mape['mape'].min()
        st.metric("最小MAPE", f"{min_mape:.2f}%")
    with col4:
        std_mape = daily_mape['mape'].std()
        st.metric("MAPE标准差", f"{std_mape:.2f}%")
    
    st.markdown("##### 日MAPE变化趋势")
    
    threshold = 8.0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_mape['date'],
        y=daily_mape['mape'],
        mode='lines+markers',
        name='日MAPE',
        line=dict(color='#1E3A5F', width=2),
        marker=dict(
            size=8,
            color=['#dc3545' if m > threshold else '#1E3A5F' for m in daily_mape['mape']]
        )
    ))
    
    fig.add_hline(
        y=threshold,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text=f'阈值 {threshold}%'
    )
    
    fig.update_layout(
        title='日MAPE变化趋势',
        xaxis_title='日期',
        yaxis_title='MAPE %',
        hovermode='x unified'
    )
    st_plot(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. 大误差日识别")
    
    large_error_days = analyzer.identify_large_error_days(result_df, threshold=threshold)
    
    if len(large_error_days) > 0:
        st.warning(f"⚠️ 识别到 {len(large_error_days)} 个大误差日（MAPE > {threshold}%）")
        
        if 'attributions' not in st.session_state:
            st.session_state.attributions = {}
        
        attribution_options = ['突发天气', '节假日效应', '特殊事件', '数据质量问题', '模型偏差', '其他']
        
        st.markdown("##### 误差归因标注")
        
        attribution_data = []
        
        for idx, row in large_error_days.iterrows():
            date_str = row['date']
            default_attr = st.session_state.attributions.get(date_str, '')
            
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                st.write(f"📅 {date_str}")
            with col2:
                st.write(f"MAPE: {row['mape']:.2f}%")
            with col3:
                attribution = st.selectbox(
                    "归因",
                    attribution_options,
                    index=attribution_options.index(default_attr) if default_attr in attribution_options else 0,
                    key=f"attr_{date_str}"
                )
                st.session_state.attributions[date_str] = attribution
            
            attribution_data.append({
                '日期': date_str,
                'MAPE': row['mape'],
                '归因': st.session_state.attributions.get(date_str, '')
            })
        
        if attribution_data:
            st.markdown("##### 大误差日归因汇总")
            attr_df = pd.DataFrame(attribution_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(attr_df, use_container_width=True)
            
            with col2:
                attr_count = pd.Series([d['归因'] for d in attribution_data]).value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=attr_count.index,
                    values=attr_count.values,
                    hole=0.4
                )])
                fig.update_layout(title='误差归因分布')
                st_plot(fig, use_container_width=True)
            
            if st.button("💾 保存归因结果"):
                try:
                    attr_save_path = os.path.join('data', 'processed', 'error_attributions.csv')
                    attr_df.to_csv(attr_save_path, index=False, encoding='utf-8-sig')
                    st.success(f"✅ 归因结果已保存至 {attr_save_path}")
                except Exception as e:
                    st.error(f"保存失败：{str(e)}")
    else:
        st.success(f"✅ 未识别到大误差日（所有日期MAPE ≤ {threshold}%）")
    
    st.markdown("---")
    st.subheader("3. 误差分布分析")
    
    result_df['error'] = result_df['predicted'] - result_df['actual']
    result_df['error_pct'] = np.abs(result_df['error'] / result_df['actual']) * 100
    result_df['date'] = result_df['datetime'].dt.date
    
    error_stats = analyzer.analyze_error_distribution(result_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("平均绝对误差", f"{error_stats['mean_abs_error']:.2f} MW")
    with col2:
        st.metric("误差标准差", f"{error_stats['std_error']:.2f} MW")
    with col3:
        st.metric("误差偏度", f"{error_stats['skewness']:.3f}")
    with col4:
        st.metric("误差峰度", f"{error_stats['kurtosis']:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_error_distribution(result_df['error'], title='绝对误差分布')
        st_plot(fig, use_container_width=True)
    
    with col2:
        fig = plot_error_distribution(result_df['error_pct'], title='相对误差(MAPE)分布')
        st_plot(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("4. 误差与气温相关性")
    
    if st.session_state.load_data is not None and 'temperature_C' in st.session_state.load_data.columns:
        temp_df = st.session_state.load_data[['temperature_C']].reset_index()
        merged_df = result_df.merge(temp_df, on='datetime', how='left')
        merged_df = merged_df.dropna(subset=['temperature_C'])
        
        correlation = analyzer.analyze_temperature_correlation(merged_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Pearson相关系数", f"{correlation['pearson']:.3f}")
            st.metric("Spearman相关系数", f"{correlation['spearman']:.3f}")
        
        with col2:
            fig = plot_error_vs_temperature(
                merged_df,
                'temperature_C',
                'error_pct',
                title='误差与气温散点图'
            )
            st_plot(fig, use_container_width=True)
        
        st.markdown("##### 不同温度区间的误差分布")
        
        temp_bins = [-np.inf, 0, 10, 20, 30, np.inf]
        temp_labels = ['<0℃', '0-10℃', '10-20℃', '20-30℃', '>30℃']
        merged_df['temp_range'] = pd.cut(merged_df['temperature_C'], bins=temp_bins, labels=temp_labels)
        
        temp_error = merged_df.groupby('temp_range')['error_pct'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=temp_error['temp_range'],
                y=temp_error['mean'],
                error_y=dict(type='data', array=temp_error['std']),
                marker_color='#38B2AC'
            )
        ])
        fig.update_layout(
            title='不同温度区间的平均MAPE',
            xaxis_title='温度区间',
            yaxis_title='平均MAPE %'
        )
        st_plot(fig, use_container_width=True)
    else:
        st.info("⚠️ 缺少气温数据，无法进行气温相关性分析")
    
    st.markdown("---")
    st.subheader("5. 误差归因模型")
    
    if len(large_error_days) > 0 and st.session_state.calendar_data is not None:
        st.info("💡 基于历史归因数据的误差原因智能分析")
        
        calendar_df = st.session_state.calendar_data.copy()
        calendar_df['date'] = pd.to_datetime(calendar_df['date']).dt.date
        
        daily_mape['date'] = pd.to_datetime(daily_mape['date']).dt.date
        merged = daily_mape.merge(calendar_df, on='date', how='left')
        
        if len(merged) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                day_type_error = merged.groupby('day_type')['mape'].agg(['mean', 'count']).reset_index()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=day_type_error['day_type'],
                        y=day_type_error['mean'],
                        marker_color=['#1E3A5F', '#38B2AC', '#ED8936']
                    )
                ])
                fig.update_layout(
                    title='不同日类型的平均MAPE',
                    xaxis_title='日类型',
                    yaxis_title='平均MAPE %'
                )
                st_plot(fig, use_container_width=True)
            
            with col2:
                holiday_error = merged.groupby('is_holiday')['mape'].agg(['mean', 'count']).reset_index()
                holiday_error['is_holiday'] = holiday_error['is_holiday'].map({True: '节假日', False: '非节假日'})
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=holiday_error['is_holiday'],
                        values=holiday_error['mean'],
                        hole=0.4
                    )
                ])
                fig.update_layout(title='节假日与非节假日平均MAPE对比')
                st_plot(fig, use_container_width=True)


def page_anomaly_detection():
    st.header("🔍 负荷异常检测与事件关联")

    if st.session_state.load_data is None:
        st.warning("⚠️ 请先在'数据导入'页面完成数据导入")
        return

    st.markdown("---")

    load_df = st.session_state.load_data.copy()
    calendar_df = st.session_state.calendar_data

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        baseline_days = st.slider("基线历史天数", 7, 30, 14, 1)
    with col2:
        mad_threshold = st.slider("MAD阈值倍数", 2.0, 5.0, 3.0, 0.5)
    with col3:
        max_gap = st.slider("事件聚合最大间隔(分钟)", 15, 90, 45, 15)
    with col4:
        corr_window = st.slider("关联搜索窗口(小时)", 1, 6, 2, 1)

    st.markdown("---")

    if st.button("🔍 执行异常检测与关联分析", type="primary") or st.session_state.anomaly_detection_result is not None:
        need_recompute = st.button("🔄 重新执行分析", key='recompute_anomaly')

        if need_recompute or st.session_state.anomaly_detection_result is None:
            with st.spinner("正在执行异常检测与事件关联分析..."):
                try:
                    detector = AnomalyDetector(
                        target_col='active_power_MW',
                        datetime_col='datetime',
                        temp_col='temperature_C',
                        solar_col='solar_irradiance_Wm2',
                        baseline_days=baseline_days,
                        mad_threshold=mad_threshold,
                        max_gap_minutes=max_gap,
                        correlation_window_hours=corr_window
                    )
                    result = detector.run_full_analysis(load_df, calendar_df)
                    st.session_state.anomaly_detection_result = result
                    st.success("✅ 异常检测与关联分析完成！")
                except Exception as e:
                    st.error(f"分析失败：{str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    return

        result = st.session_state.anomaly_detection_result
        if result is None:
            return

        anomaly_events = result['anomaly_events']
        correlation_stats = result['correlation_stats']
        monthly_stats = result['monthly_stats']
        anomaly_points_df = result['anomaly_points']

        st.subheader("1. 异常事件概览")

        total_events = len(anomaly_events)
        if total_events > 0:
            pos_count = len(anomaly_events[anomaly_events['deviation_direction'].str.contains('正')])
            neg_count = total_events - pos_count
            avg_duration = anomaly_events['duration_minutes'].mean()
            avg_deviation = anomaly_events['peak_deviation_ratio'].abs().mean()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("异常事件总数", total_events)
            with col2:
                st.metric("正异常(突增)", pos_count)
            with col3:
                st.metric("负异常(骤降)", neg_count)
            with col4:
                st.metric("平均持续时长", f"{avg_duration:.0f} 分钟")
        else:
            st.info("未检测到异常事件")

        st.markdown("---")
        st.subheader("2. 异常事件时间轴")

        if total_events > 0:
            color_map = {
                '气象关联': '#dc3545',
                '日历关联': '#ED8936',
                '复合关联(气象+日历)': '#805AD5',
                '未知原因': '#718096'
            }

            scatter_df = anomaly_events.copy()
            scatter_df['color'] = scatter_df['correlation_type'].map(color_map)
            scatter_df['abs_deviation'] = scatter_df['peak_deviation_ratio'].abs()

            fig = go.Figure()

            for corr_type, color in color_map.items():
                sub = scatter_df[scatter_df['correlation_type'] == corr_type]
                if len(sub) > 0:
                    fig.add_trace(go.Scatter(
                        x=sub['start_time'],
                        y=sub['peak_deviation_ratio'],
                        mode='markers',
                        name=corr_type,
                        marker=dict(
                            size=10,
                            color=color,
                            opacity=0.8,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=(
                            '开始时间: %{x}<br>'
                            '偏离倍数: %{y:.2f}<br>'
                            '持续时长: %{customdata[0]}分钟<br>'
                            '关联类型: ' + corr_type + '<br>'
                            '<extra></extra>'
                        ),
                        customdata=sub[['duration_minutes']].values
                    ))

            fig.add_hline(
                y=mad_threshold,
                line_dash='dash',
                line_color='red',
                opacity=0.5,
                annotation_text=f'+{mad_threshold}σ (阈值)'
            )
            fig.add_hline(
                y=-mad_threshold,
                line_dash='dash',
                line_color='red',
                opacity=0.5,
                annotation_text=f'-{mad_threshold}σ (阈值)'
            )

            fig.update_layout(
                title='异常事件时间轴（偏离倍数）',
                xaxis_title='时间',
                yaxis_title='偏离倍数 (σ)',
                hovermode='x unified',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                template='plotly_white',
                height=450
            )
            st_plot(fig, use_container_width=True)
        else:
            st.info("暂无异常事件可展示")

        st.markdown("---")
        st.subheader("3. 事件详情与关联分析")

        if total_events > 0:
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("##### 异常事件列表")

                display_df = anomaly_events[[
                    'event_id', 'start_time', 'end_time', 'duration_minutes',
                    'peak_deviation_ratio', 'deviation_direction',
                    'correlation_type', 'weather_event_types', 'calendar_info'
                ]].copy()

                display_df = display_df.rename(columns={
                    'event_id': '事件ID',
                    'start_time': '开始时间',
                    'end_time': '结束时间',
                    'duration_minutes': '持续时长(分钟)',
                    'peak_deviation_ratio': '峰值偏离倍数',
                    'deviation_direction': '偏离方向',
                    'correlation_type': '关联类型',
                    'weather_event_types': '关联气象',
                    'calendar_info': '关联日历'
                })

                sort_options = ['开始时间', '峰值偏离倍数', '持续时长(分钟)']
                sort_by = st.selectbox(
                    "按列排序",
                    sort_options,
                    index=0,
                    key='event_sort'
                )
                sort_asc = st.checkbox("升序", value=False, key='event_sort_asc')

                if sort_by == '开始时间':
                    display_df = display_df.sort_values('开始时间', ascending=sort_asc)
                elif sort_by == '峰值偏离倍数':
                    display_df = display_df.sort_values('峰值偏离倍数', ascending=sort_asc)
                else:
                    display_df = display_df.sort_values('持续时长(分钟)', ascending=sort_asc)

                st.dataframe(
                    display_df.style.format({
                        '峰值偏离倍数': '{:.2f}'
                    }),
                    use_container_width=True,
                    height=450
                )

            with col_right:
                st.markdown("##### 关联类型分布")

                if correlation_stats is not None and len(correlation_stats) > 0:
                    pie_colors = [color_map.get(t, '#718096') for t in correlation_stats['关联类型']]

                    fig = go.Figure(data=[go.Pie(
                        labels=correlation_stats['关联类型'],
                        values=correlation_stats['事件数量'],
                        hole=0.45,
                        marker=dict(colors=pie_colors),
                        textinfo='label+percent',
                        textfont=dict(size=11),
                        hovertemplate=(
                            '%{label}<br>'
                            '事件数: %{value}<br>'
                            '占比: %{percent}<extra></extra>'
                        )
                    )])
                    fig.update_layout(
                        title='异常事件关联类型占比',
                        template='plotly_white',
                        height=280,
                        showlegend=False
                    )
                    st_plot(fig, use_container_width=True)
                else:
                    st.info("暂无关联统计数据")

                st.markdown("##### 月度异常频次")

                if monthly_stats is not None and len(monthly_stats) > 0:
                    fig = go.Figure(data=[go.Bar(
                        x=monthly_stats['month'],
                        y=monthly_stats['异常事件数'],
                        marker_color='#1E3A5F',
                        text=monthly_stats['异常事件数'],
                        textposition='outside',
                        hovertemplate='月份: %{x}<br>异常事件数: %{y}<extra></extra>'
                    )])
                    fig.update_layout(
                        title='各月异常事件数量',
                        xaxis_title='月份',
                        yaxis_title='异常事件数',
                        template='plotly_white',
                        height=280
                    )
                    st_plot(fig, use_container_width=True)
                else:
                    st.info("暂无月度统计数据")

            if correlation_stats is not None and len(correlation_stats) > 0:
                st.markdown("---")
                st.subheader("4. 关联分析统计")

                col1, col2, col3, col4 = st.columns(4)
                for i, (_, row) in enumerate(correlation_stats.iterrows()):
                    if i < 4:
                        with [col1, col2, col3, col4][i]:
                            st.metric(
                                row['关联类型'],
                                f"{int(row['事件数量'])} 件",
                                f"{row['占比']*100:.1f}%"
                            )
        else:
            st.info("未检测到异常事件，建议调整检测参数后重试")


def main():
    st.sidebar.title("⚡ 区域电网负荷预测与需求响应优化分析系统")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "选择功能模块",
        ["📊 数据导入", "📈 短期预测", "⚡ 超短期预测", 
         "📊 峰谷分析", "🔧 需求响应", "📉 误差分析", "🔍 异常检测"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    💡 使用说明：
    
    1. 先在「数据导入」页面上传数据并生成特征
    2. 在「短期预测」页面训练预测模型
    3. 在「峰谷分析」页面分析负荷特性
    4. 在「需求响应」页面模拟优化调度
    5. 在「误差分析」页面评估预测精度
    6. 在「异常检测」页面分析负荷异常与关联因素
    """)
    
    if page == "📊 数据导入":
        page_data_import()
    elif page == "📈 短期预测":
        page_short_term_forecast()
    elif page == "⚡ 超短期预测":
        page_ultra_short_forecast()
    elif page == "📊 峰谷分析":
        page_peak_valley_analysis()
    elif page == "🔧 需求响应":
        page_demand_response()
    elif page == "📉 误差分析":
        page_error_analysis()
    elif page == "🔍 异常检测":
        page_anomaly_detection()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**版本**: v1.0.0")
    st.sidebar.markdown("**更新时间**: 2026-06")


if __name__ == '__main__':
    main()