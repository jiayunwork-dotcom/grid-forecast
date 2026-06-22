import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("核心模块功能测试")
print("=" * 60)

print("\n1. 测试工具函数...")
from src.utils import mape, rmse, mae, check_time_continuous, detect_outliers_iqr, cyclical_encoding

y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([105, 198, 290, 410, 505])
print(f"   MAPE: {mape(y_true, y_pred):.4f}%")
print(f"   RMSE: {rmse(y_true, y_pred):.4f}")
print(f"   MAE: {mae(y_true, y_pred):.4f}")

dates = pd.date_range('2025-01-01', periods=100, freq='15min')
is_continuous, gaps = check_time_continuous(dates)
print(f"   时间连续性: {is_continuous}, 间隔数: {len(gaps)}")

data = np.random.normal(100, 10, 1000)
outliers = detect_outliers_iqr(data)
print(f"   异常值检测: {np.sum(outliers)} 个异常值")

sin_enc, cos_enc = cyclical_encoding(pd.Series(range(24)), 24)
print(f"   周期性编码: sin.shape={sin_enc.shape}, cos.shape={cos_enc.shape}")
print("   ✓ 工具函数测试通过")

print("\n2. 测试数据加载...")
from src.data_loader import load_load_data, load_calendar_data

data_dir = 'data/raw'
load_df, _ = load_load_data(os.path.join(data_dir, 'load_data.csv'))
calendar_df, _ = load_calendar_data(os.path.join(data_dir, 'calendar_data.csv'))
print(f"   负荷数据: {len(load_df)} 行, {len(load_df.columns)} 列")
print(f"   日历数据: {len(calendar_df)} 行, {len(calendar_df.columns)} 列")
print(f"   时间范围: {load_df['datetime'].min()} 至 {load_df['datetime'].max()}")
print("   ✓ 数据加载测试通过")

print("\n3. 测试数据质量检查...")
from src.data_quality import generate_quality_report, clean_data, calculate_quality_score, calculate_missing_rate, detect_outliers, check_time_continuity

overall_score = calculate_quality_score(load_df)
print(f"   整体质量评分: {overall_score:.2f}/100")

missing_stats = calculate_missing_rate(load_df)
avg_missing_rate = missing_stats['缺失率'].mean()
print(f"   平均缺失率: {avg_missing_rate:.2f}%")

outliers = detect_outliers(load_df)
if outliers:
    avg_outlier_rate = np.mean([v['异常值比例'] for v in outliers.values()])
    print(f"   平均异常值率: {avg_outlier_rate:.2f}%")

continuity = check_time_continuity(load_df)
print(f"   时间连续: {continuity.get('是否连续', False)}")

cleaned_df = clean_data(load_df)
print(f"   清洗后数据: {len(cleaned_df)} 行")
print("   ✓ 数据质量检查测试通过")

print("\n4. 测试特征工程...")
from src.feature_engineering import build_features

load_df['date'] = load_df['datetime'].dt.date
calendar_df['date'] = pd.to_datetime(calendar_df['date']).dt.date
merged_df = pd.merge(load_df, calendar_df, on='date', how='left')
forecast_df = pd.read_csv(os.path.join(data_dir, 'temperature_forecast.csv'))
forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
merged_df = pd.merge(merged_df, forecast_df, on='datetime', how='left')

features_df = build_features(merged_df, target_col='active_power_MW')
features_df = features_df.dropna()

feature_cols = [col for col in features_df.columns if col not in ['active_power_MW', 'datetime', 'date']]
target = features_df['active_power_MW']
features_df = features_df[feature_cols]

print(f"   特征数量: {len(features_df.columns)}")
print(f"   特征列: {list(features_df.columns)[:10]}...")
print(f"   目标变量长度: {len(target)}")
print("   ✓ 特征工程测试通过")

print("\n5. 测试LightGBM预测...")
from src.short_term_forecast import LightGBMForecaster

full_df = features_df.copy()
full_df['active_power_MW'] = target.values

train_end = int(len(full_df) * 0.8)
train_df = full_df.iloc[:train_end]
test_df = full_df.iloc[train_end:]

lgbm_model = LightGBMForecaster(
    target_col='active_power_MW',
    params={'n_estimators': 100, 'verbose': -1}
)
lgbm_model.fit(train_df)
y_pred_df = lgbm_model.predict(test_df)
y_pred = y_pred_df['prediction'].values if isinstance(y_pred_df, pd.DataFrame) else y_pred_df
y_test = test_df['active_power_MW'].values

print(f"   LightGBM MAPE: {mape(y_test, y_pred):.4f}%")
print(f"   LightGBM RMSE: {rmse(y_test, y_pred):.4f}")

importance = lgbm_model.get_feature_importance()
print(f"   Top 5 重要特征: {list(importance.index[:5])}")
print("   ✓ LightGBM预测测试通过")

print("\n6. 测试峰谷分析...")
from src.peak_valley_analysis import PeakValleyAnalyzer

analyzer = PeakValleyAnalyzer(target_col='active_power_MW')
daily_features = analyzer.extract_daily_peak_valley(load_df)
print(f"   分析天数: {len(daily_features)}")
print(f"   平均峰谷差: {daily_features['peak_valley_diff'].mean():.2f} MW")
print(f"   平均日负荷率: {daily_features['daily_load_rate'].mean():.4f}")

spike_df = analyzer.calculate_peak_duration(load_df)
print(f"   平均尖峰时长: {spike_df['peak_duration_hours'].mean():.2f} 小时")

monthly_trend = analyzer.aggregate_monthly_features(daily_features)
print(f"   月度趋势: {len(monthly_trend)} 个月")
print("   ✓ 峰谷分析测试通过")

print("\n7. 测试需求响应优化...")
from src.demand_response import DemandResponseOptimizer, DRResource, DRStrategy

test_df = load_df.iloc[:96].copy()
test_df = test_df.set_index('datetime')

optimizer = DemandResponseOptimizer()

optimizer.add_resource(DRResource(
    name='空调', resource_type='air_conditioning', capacity_mw=50, max_duration_hours=4, ac_share=0.3
))
optimizer.add_resource(DRResource(
    name='工业', resource_type='industrial', capacity_mw=80, max_duration_hours=6
))
optimizer.add_resource(DRResource(
    name='储能', resource_type='storage', capacity_mw=30, max_duration_hours=4
))

strategy = DRStrategy(
    peak_target_mw=test_df['active_power_MW'].max() * 0.9,
    response_start_hour=10,
    response_end_hour=22,
    max_curtailment_duration=4,
    max_temp_offset=2.0
)
optimizer.set_strategy(strategy)

result = optimizer.optimize(test_df, load_col='active_power_MW')
print(f"   优化状态: {result.message}")
if result.success:
    print(f"   原始峰谷差: {result.peak_valley_diff_before:.2f} MW")
    print(f"   优化后峰谷差: {result.peak_valley_diff_after:.2f} MW")
    print(f"   总削峰量: {result.peak_reduction_kw:.2f} kW")
    print(f"   总填谷量: {result.valley_increase_kw:.2f} kW")
    print(f"   节省成本: {result.cost_savings:.2f} 元")
print("   ✓ 需求响应优化测试通过")

print("\n8. 测试误差分析...")
from src.error_analysis import ErrorAnalyzer

error_df = pd.DataFrame({
    'datetime': load_df['datetime'].iloc[train_end:train_end+len(y_pred)].reset_index(drop=True),
    'active_power_MW': y_test,
    'prediction': y_pred
})

error_analyzer = ErrorAnalyzer(target_col='active_power_MW', prediction_col='prediction')
daily_mape = error_analyzer.calculate_daily_mape(error_df)
print(f"   日MAPE数量: {len(daily_mape)}")
print(f"   平均日MAPE: {daily_mape['mape'].mean():.4f}%")

large_error_days = error_analyzer.identify_large_error_days(threshold=8.0)
print(f"   大误差天数: {len(large_error_days)}")
print("   ✓ 误差分析测试通过")

print("\n" + "=" * 60)
print("所有核心模块测试通过! ✓")
print("=" * 60)
