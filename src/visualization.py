import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Optional, Dict, Any, Union
from plotly.graph_objects import Figure


def plot_time_series(df: pd.DataFrame, datetime_col: str, value_cols: Union[str, List[str]], title: str = '时间序列图', xlabel: str = '时间', ylabel: str = '数值') -> Figure:
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    fig = go.Figure()
    
    for col in value_cols:
        fig.add_trace(go.Scatter(x=df[datetime_col], y=df[col], mode='lines', name=col, hovertemplate=f'{col}: %{{y:.2f}}<br>时间: %{{x}}<extra></extra>'))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), template='plotly_white')
    
    return fig


def plot_prediction_comparison(df: pd.DataFrame, datetime_col: str, actual_col: str, pred_col: str, lower_col: Optional[str] = None, upper_col: Optional[str] = None, title: str = '预测结果对比', xlabel: str = '时间', ylabel: str = '负荷') -> Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df[datetime_col], y=df[actual_col], mode='lines', name='实际值', line=dict(color='blue', width=2), hovertemplate='实际值: %{y:.2f}<br>时间: %{x}<extra></extra>'))
    
    fig.add_trace(go.Scatter(x=df[datetime_col], y=df[pred_col], mode='lines', name='预测值', line=dict(color='red', width=2, dash='dash'), hovertemplate='预测值: %{y:.2f}<br>时间: %{x}<extra></extra>'))
    
    if lower_col is not None and upper_col is not None:
        fig.add_trace(go.Scatter(x=df[datetime_col], y=df[upper_col], mode='lines', line=dict(color='rgba(255, 0, 0, 0.2)'), name='置信区间上限', showlegend=False))
        
        fig.add_trace(go.Scatter(x=df[datetime_col], y=df[lower_col], mode='lines', line=dict(color='rgba(255, 0, 0, 0.2)'), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='95%置信区间'))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), template='plotly_white')
    
    return fig


def plot_feature_importance(feature_names: List[str], importance: np.ndarray, top_k: int = 20, title: str = '特征重要性') -> Figure:
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=True).tail(top_k)
    
    fig = go.Figure(go.Bar(x=importance_df['importance'], y=importance_df['feature'], orientation='h', text=importance_df['importance'].round(4), textposition='outside', hovertemplate='特征: %{y}<br>重要性: %{x:.4f}<extra></extra>'))
    
    fig.update_layout(title=title, xaxis_title='重要性', yaxis_title='特征', height=max(400, top_k * 25), template='plotly_white')
    
    return fig


def plot_error_distribution(errors: np.ndarray, bins: int = 50, title: str = '误差分布直方图', xlabel: str = '误差值', ylabel: str = '频数') -> Figure:
    errors = np.array(errors).flatten()
    
    fig = go.Figure(go.Histogram(x=errors, nbinsx=bins, marker=dict(color='rgba(55, 83, 109, 0.7)', line=dict(color='rgba(55, 83, 109, 1)', width=1)), hovertemplate='误差区间: %{x}<br>频数: %{y}<extra></extra>'))
    
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    fig.add_vline(x=mean_err, line_dash='dash', line_color='red', annotation_text=f'均值: {mean_err:.4f}', annotation_position='top right')
    
    fig.add_vline(x=mean_err + std_err, line_dash='dot', line_color='orange', annotation_text=f'+σ: {mean_err + std_err:.4f}', annotation_position='top right')
    
    fig.add_vline(x=mean_err - std_err, line_dash='dot', line_color='orange', annotation_text=f'-σ: {mean_err - std_err:.4f}', annotation_position='top left')
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template='plotly_white')
    
    return fig


def plot_error_vs_temperature(errors: np.ndarray, temperature: np.ndarray, title: str = '误差-气温散点图', xlabel: str = '气温 (°C)', ylabel: str = '误差') -> Figure:
    errors = np.array(errors).flatten()
    temperature = np.array(temperature).flatten()
    
    valid_mask = ~np.isnan(errors) & ~np.isnan(temperature)
    errors = errors[valid_mask]
    temperature = temperature[valid_mask]
    
    fig = go.Figure(go.Scatter(x=temperature, y=errors, mode='markers', marker=dict(size=6, color=errors, colorscale='RdBu_r', showscale=True, colorbar=dict(title='误差')), hovertemplate='气温: %{x:.1f}°C<br>误差: %{y:.4f}<extra></extra>'))
    
    z = np.polyfit(temperature, errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(temperature.min(), temperature.max(), 100)
    
    fig.add_trace(go.Scatter(x=x_trend, y=p(x_trend), mode='lines', name=f'趋势线 (y={z[0]:.4f}x+{z[1]:.4f})', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template='plotly_white', showlegend=True)
    
    return fig


def plot_peak_valley(df: pd.DataFrame, datetime_col: str, value_col: str, peak_indices: Optional[np.ndarray] = None, valley_indices: Optional[np.ndarray] = None, title: str = '峰谷标注曲线图', xlabel: str = '时间', ylabel: str = '负荷') -> Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df[datetime_col], y=df[value_col], mode='lines', name=value_col, line=dict(color='blue', width=2), hovertemplate=f'{value_col}: %{{y:.2f}}<br>时间: %{{x}}<extra></extra>'))
    
    if peak_indices is not None:
        peak_times = df.iloc[peak_indices][datetime_col]
        peak_values = df.iloc[peak_indices][value_col]
        fig.add_trace(go.Scatter(x=peak_times, y=peak_values, mode='markers', name='峰值', marker=dict(color='red', size=10, symbol='triangle-up'), hovertemplate='峰值: %{y:.2f}<br>时间: %{x}<extra></extra>'))
    
    if valley_indices is not None:
        valley_times = df.iloc[valley_indices][datetime_col]
        valley_values = df.iloc[valley_indices][value_col]
        fig.add_trace(go.Scatter(x=valley_times, y=valley_values, mode='markers', name='谷值', marker=dict(color='green', size=10, symbol='triangle-down'), hovertemplate='谷值: %{y:.2f}<br>时间: %{x}<extra></extra>'))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), template='plotly_white')
    
    return fig


def plot_dispatch_comparison(df: pd.DataFrame, datetime_col: str, base_cols: List[str], optimized_cols: List[str], title: str = '调度前后对比图', xlabel: str = '时间', ylabel: str = '出力') -> Figure:
    fig = go.Figure()
    
    colors_optimized = px.colors.qualitative.Plotly
    colors_base = [f'rgba{tuple(int(c[1:][i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}' for c in colors_optimized[:len(base_cols)]]
    
    for i, col in enumerate(base_cols):
        fig.add_trace(go.Scatter(x=df[datetime_col], y=df[col], mode='lines', stackgroup='base', name=f'调度前-{col}', line=dict(width=0.5), fillcolor=colors_base[i % len(colors_base)], hovertemplate=f'调度前-{col}: %{{y:.2f}}<br>时间: %{{x}}<extra></extra>'))
    
    for i, col in enumerate(optimized_cols):
        fig.add_trace(go.Scatter(x=df[datetime_col], y=df[col], mode='lines', stackgroup='optimized', name=f'调度后-{col}', line=dict(width=0.5), fillcolor=colors_optimized[i % len(colors_optimized)], hovertemplate=f'调度后-{col}: %{{y:.2f}}<br>时间: %{{x}}<extra></extra>'))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), template='plotly_white')
    
    return fig


def plot_resource_stack(df: pd.DataFrame, datetime_col: str, resource_cols: List[str], title: str = '资源出力堆叠图', xlabel: str = '时间', ylabel: str = '出力') -> Figure:
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, col in enumerate(resource_cols):
        fig.add_trace(go.Scatter(x=df[datetime_col], y=df[col], mode='lines', stackgroup='one', name=col, line=dict(width=0.5), fillcolor=colors[i % len(colors)], hovertemplate=f'{col}: %{{y:.2f}}<br>时间: %{{x}}<extra></extra>'))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), template='plotly_white')
    
    return fig


def plot_missing_heatmap(df: pd.DataFrame, title: str = '缺失率热力图') -> Figure:
    missing_rate = df.isnull().mean() * 100
    
    fig = go.Figure(go.Heatmap(z=[missing_rate.values], x=missing_rate.index, y=['缺失率(%)'], colorscale='Reds', text=[[f'{v:.2f}%' for v in missing_rate.values]], texttemplate='%{text}', hovertemplate='列: %{x}<br>缺失率: %{z:.2f}%<extra></extra>'))
    
    fig.update_layout(title=title, height=200, template='plotly_white')
    
    return fig


def st_plot(fig: Figure, use_container_width: bool = True) -> None:
    try:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=use_container_width)
    except ImportError:
        print('Streamlit not available. Use fig.show() to display the plot.')
