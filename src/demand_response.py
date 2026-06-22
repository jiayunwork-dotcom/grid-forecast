import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class DRResource:
    name: str
    resource_type: str
    capacity_mw: float
    max_duration_hours: float = 4.0
    min_stay_hours: float = 1.0
    ramp_rate_mw_per_min: float = 0.5
    cost_per_mwh: float = 0.0
    ac_share: float = 0.0


@dataclass
class DRStrategy:
    peak_target_mw: Optional[float] = None
    valley_target_mw: Optional[float] = None
    response_start_hour: int = 10
    response_end_hour: int = 22
    max_curtailment_duration: float = 4.0
    max_temp_offset: float = 2.0
    min_temp_offset: float = -2.0
    total_energy_conservation: bool = True


@dataclass
class DRResult:
    success: bool
    message: str
    base_load: np.ndarray
    optimized_load: np.ndarray
    ac_curtailment: np.ndarray
    industrial_curtailment: np.ndarray
    storage_charge: np.ndarray
    storage_discharge: np.ndarray
    peak_reduction_kw: float
    valley_increase_kw: float
    peak_valley_diff_before: float
    peak_valley_diff_after: float
    cost_savings: float
    scheduling_df: Optional[pd.DataFrame] = None


class DemandResponseOptimizer:
    def __init__(self):
        self.resources: List[DRResource] = []
        self.strategy: DRStrategy = DRStrategy()
        self.time_index: Optional[pd.DatetimeIndex] = None

    def add_resource(self, resource: DRResource) -> None:
        self.resources.append(resource)

    def set_strategy(self, strategy: DRStrategy) -> None:
        self.strategy = strategy

    def _build_ac_curtailment(self, n_timepoints: int, dt_minutes: int,
                              ac_capacity_mw: float, base_load: np.ndarray,
                              ac_share: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ac_load = base_load * ac_share
        max_curtail = ac_capacity_mw * np.ones(n_timepoints)
        
        max_consecutive = int(self.strategy.max_curtailment_duration * 60 / dt_minutes)
        
        hours = self.time_index.hour
        in_response_window = (hours >= self.strategy.response_start_hour) & \
                            (hours <= self.strategy.response_end_hour)
        max_curtail[~in_response_window] = 0
        
        return ac_load, max_curtail, in_response_window.astype(int)

    def _build_industrial_curtailment(self, n_timepoints: int, dt_minutes: int,
                                      ind_capacity_mw: float) -> np.ndarray:
        max_curtail = ind_capacity_mw * np.ones(n_timepoints)
        
        hours = self.time_index.hour
        in_response_window = (hours >= self.strategy.response_start_hour) & \
                            (hours <= self.strategy.response_end_hour)
        max_curtail[~in_response_window] = 0
        
        return max_curtail

    def _build_storage(self, n_timepoints: int, dt_minutes: int,
                       power_mw: float, capacity_mwh: float
                       ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        dt_hours = dt_minutes / 60.0
        max_charge = power_mw * np.ones(n_timepoints)
        max_discharge = power_mw * np.ones(n_timepoints)
        
        soc_min = 0.1 * capacity_mwh
        soc_max = 0.9 * capacity_mwh
        soc_initial = 0.5 * capacity_mwh
        
        return max_charge, max_discharge, soc_min, soc_max, soc_initial, dt_hours

    def _temp_constraint(self, ac_curtailment: np.ndarray, ac_load: np.ndarray,
                         dt_hours: float) -> np.ndarray:
        temp_impact = np.cumsum(ac_curtailment / (ac_load + 1e-6)) * dt_hours * 0.5
        return temp_impact

    def optimize(self, base_load_df: pd.DataFrame, load_col: str = 'active_power_MW',
                 peak_price: float = 1.2, valley_price: float = 0.4) -> DRResult:
        if self.time_index is None:
            self.time_index = base_load_df.index
        
        base_load = base_load_df[load_col].values.astype(np.float64)
        n_timepoints = len(base_load)
        dt_minutes = (self.time_index[1] - self.time_index[0]).total_seconds() / 60
        dt_hours = dt_minutes / 60.0
        
        ac_resource = next((r for r in self.resources if r.resource_type == 'air_conditioning'), None)
        ind_resource = next((r for r in self.resources if r.resource_type == 'industrial'), None)
        storage_resource = next((r for r in self.resources if r.resource_type == 'storage'), None)
        
        n_vars = 0
        var_info = []
        
        if ac_resource:
            n_vars += n_timepoints
            var_info.append(('ac', ac_resource))
        
        if ind_resource:
            n_vars += n_timepoints
            var_info.append(('ind', ind_resource))
        
        if storage_resource:
            n_vars += 2 * n_timepoints
            var_info.append(('storage_charge', storage_resource))
            var_info.append(('storage_discharge', storage_resource))
        
        n_vars += 2
        var_info.append(('peak_var', None))
        var_info.append(('valley_var', None))
        
        peak_before = np.max(base_load)
        valley_before = np.min(base_load)
        diff_before = peak_before - valley_before
        
        c = np.zeros(n_vars)
        c[-2] = 1.0
        c[-1] = -1.0
        
        bounds = []
        
        ac_load = np.zeros(n_timepoints)
        max_ac_curtail = np.zeros(n_timepoints)
        if ac_resource:
            ac_load, max_ac_curtail, in_window = self._build_ac_curtailment(
                n_timepoints, dt_minutes, ac_resource.capacity_mw, base_load, ac_resource.ac_share
            )
            for i in range(n_timepoints):
                bounds.append((0, max_ac_curtail[i]))
        
        max_ind_curtail = np.zeros(n_timepoints)
        if ind_resource:
            max_ind_curtail = self._build_industrial_curtailment(
                n_timepoints, dt_minutes, ind_resource.capacity_mw
            )
            for i in range(n_timepoints):
                bounds.append((0, max_ind_curtail[i]))
        
        max_charge = np.zeros(n_timepoints)
        max_discharge = np.zeros(n_timepoints)
        soc_min = soc_max = soc_initial = 0
        if storage_resource:
            max_charge, max_discharge, soc_min, soc_max, soc_initial, _ = self._build_storage(
                n_timepoints, dt_minutes, storage_resource.capacity_mw,
                storage_resource.capacity_mw
            )
            for i in range(n_timepoints):
                bounds.append((0, max_charge[i]))
            for i in range(n_timepoints):
                bounds.append((0, max_discharge[i]))
        
        bounds.append((None, None))
        bounds.append((None, None))
        
        A_ub = []
        b_ub = []
        
        var_offset = 0
        
        if ac_resource:
            for t in range(n_timepoints):
                row = np.zeros(n_vars)
                row[t] = 1.0
                row[-2] = -1.0
                if ind_resource:
                    row[n_timepoints + t] = 1.0
                if storage_resource:
                    row[2 * n_timepoints + t] = -1.0
                    row[3 * n_timepoints + t] = 1.0
                A_ub.append(row)
                b_ub.append(base_load[t] - valley_before)
            
            for t in range(n_timepoints):
                row = np.zeros(n_vars)
                row[t] = -1.0
                row[-1] = 1.0
                if ind_resource:
                    row[n_timepoints + t] = -1.0
                if storage_resource:
                    row[2 * n_timepoints + t] = 1.0
                    row[3 * n_timepoints + t] = -1.0
                A_ub.append(row)
                b_ub.append(-base_load[t] + peak_before)
            
            var_offset += n_timepoints
        
        if ind_resource:
            var_offset += n_timepoints
        
        if storage_resource:
            for t in range(n_timepoints):
                row = np.zeros(n_vars)
                for k in range(t + 1):
                    row[var_offset + k] = dt_hours
                    row[var_offset + n_timepoints + k] = -dt_hours
                A_ub.append(row)
                b_ub.append(soc_max - soc_initial)
                
                row = np.zeros(n_vars)
                for k in range(t + 1):
                    row[var_offset + k] = -dt_hours
                    row[var_offset + n_timepoints + k] = dt_hours
                A_ub.append(row)
                b_ub.append(soc_initial - soc_min)
            
            var_offset += 2 * n_timepoints
        
        A_eq = []
        b_eq = []
        
        if self.strategy.total_energy_conservation:
            row = np.zeros(n_vars)
            for t in range(n_timepoints):
                if ac_resource:
                    row[t] = 1
                if ind_resource:
                    row[n_timepoints + t] = 1
                if storage_resource:
                    row[2 * n_timepoints + t] = -1
                    row[3 * n_timepoints + t] = 1
            A_eq.append(row)
            b_eq.append(0.0)
        
        if self.strategy.peak_target_mw:
            row = np.zeros(n_vars)
            row[-2] = 1
            A_ub.append(row)
            b_ub.append(self.strategy.peak_target_mw)
        
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        
        try:
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'maxiter': 5000, 'presolve': True}
            )
            
            if not result.success:
                result = linprog(
                    c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method='highs-ipm',
                    options={'maxiter': 10000, 'presolve': True}
                )
        except Exception as e:
            return DRResult(
                success=False,
                message=f"优化求解失败: {str(e)}",
                base_load=base_load,
                optimized_load=base_load,
                ac_curtailment=np.zeros(n_timepoints),
                industrial_curtailment=np.zeros(n_timepoints),
                storage_charge=np.zeros(n_timepoints),
                storage_discharge=np.zeros(n_timepoints),
                peak_reduction_kw=0,
                valley_increase_kw=0,
                peak_valley_diff_before=diff_before,
                peak_valley_diff_after=diff_before,
                cost_savings=0
            )
        
        x = result.x
        
        ac_curtail = np.zeros(n_timepoints)
        ind_curtail = np.zeros(n_timepoints)
        storage_c = np.zeros(n_timepoints)
        storage_d = np.zeros(n_timepoints)
        
        offset = 0
        if ac_resource:
            ac_curtail = x[offset:offset + n_timepoints]
            offset += n_timepoints
        if ind_resource:
            ind_curtail = x[offset:offset + n_timepoints]
            offset += n_timepoints
        if storage_resource:
            storage_c = x[offset:offset + n_timepoints]
            storage_d = x[offset + n_timepoints:offset + 2 * n_timepoints]
        
        ac_curtail = np.maximum(0, np.minimum(ac_curtail, max_ac_curtail))
        ind_curtail = np.maximum(0, np.minimum(ind_curtail, max_ind_curtail))
        storage_c = np.maximum(0, np.minimum(storage_c, max_charge))
        storage_d = np.maximum(0, np.minimum(storage_d, max_discharge))
        
        optimized_load = base_load - ac_curtail - ind_curtail - storage_c + storage_d
        
        peak_after = np.max(optimized_load)
        valley_after = np.min(optimized_load)
        diff_after = peak_after - valley_after
        
        peak_reduction = (peak_before - peak_after) * 1000
        valley_increase = (valley_after - valley_before) * 1000
        
        hours_arr = self.time_index.hour
        peak_hours = (hours_arr >= 8) & (hours_arr <= 22)
        valley_hours = ~peak_hours
        
        energy_peak_before = np.sum(base_load[peak_hours]) * dt_hours
        energy_valley_before = np.sum(base_load[valley_hours]) * dt_hours
        energy_peak_after = np.sum(optimized_load[peak_hours]) * dt_hours
        energy_valley_after = np.sum(optimized_load[valley_hours]) * dt_hours
        
        cost_before = energy_peak_before * peak_price + energy_valley_before * valley_price
        cost_after = energy_peak_after * peak_price + energy_valley_after * valley_price
        
        total_curtail = np.sum(ac_curtail + ind_curtail) * dt_hours
        if ac_resource:
            cost_after += total_curtail * ac_resource.cost_per_mwh
        
        cost_savings = cost_before - cost_after
        
        scheduling_df = pd.DataFrame({
            'datetime': self.time_index,
            'base_load_MW': base_load,
            'optimized_load_MW': optimized_load,
            'ac_curtailment_MW': ac_curtail,
            'industrial_curtailment_MW': ind_curtail,
            'storage_charge_MW': storage_c,
            'storage_discharge_MW': storage_d,
            'load_delta_MW': optimized_load - base_load
        })
        
        return DRResult(
            success=result.success,
            message=result.message,
            base_load=base_load,
            optimized_load=optimized_load,
            ac_curtailment=ac_curtail,
            industrial_curtailment=ind_curtail,
            storage_charge=storage_c,
            storage_discharge=storage_d,
            peak_reduction_kw=peak_reduction,
            valley_increase_kw=valley_increase,
            peak_valley_diff_before=diff_before,
            peak_valley_diff_after=diff_after,
            cost_savings=cost_savings,
            scheduling_df=scheduling_df
        )


def create_default_resources(
    ac_share: float = 0.3,
    ac_capacity_mw: float = 50.0,
    industrial_capacity_mw: float = 30.0,
    storage_capacity_mwh: float = 100.0,
    storage_power_mw: float = 20.0
) -> List[DRResource]:
    resources = []
    
    resources.append(DRResource(
        name='空调负荷',
        resource_type='air_conditioning',
        capacity_mw=ac_capacity_mw,
        max_duration_hours=4.0,
        cost_per_mwh=50.0,
        ac_share=ac_share
    ))
    
    resources.append(DRResource(
        name='工业可中断负荷',
        resource_type='industrial',
        capacity_mw=industrial_capacity_mw,
        max_duration_hours=2.0,
        cost_per_mwh=80.0,
        ac_share=0.0
    ))
    
    resources.append(DRResource(
        name='分布式储能',
        resource_type='storage',
        capacity_mw=storage_power_mw,
        max_duration_hours=storage_capacity_mwh / storage_power_mw if storage_power_mw > 0 else 4,
        cost_per_mwh=30.0,
        ac_share=0.0
    ))
    
    return resources


def analyze_dr_potential(load_df: pd.DataFrame, load_col: str = 'active_power_MW',
                          peak_threshold: float = 0.8) -> Dict[str, Any]:
    load = load_df[load_col].values
    peak_val = np.max(load)
    valley_val = np.min(load)
    mean_val = np.mean(load)
    
    peak_hours_count = np.sum(load > peak_threshold * peak_val) * 15 / 60
    load_rate = mean_val / peak_val
    
    dr_potential_mw = peak_val - valley_val
    dr_potential_mwh = dr_potential_mw * peak_hours_count
    
    temp_df = load_df.copy()
    if 'datetime' in temp_df.columns and temp_df.index.name != 'datetime':
        temp_df = temp_df.set_index('datetime')
    
    if isinstance(temp_df.index, pd.DatetimeIndex):
        hourly_profile = temp_df[load_col].resample('h').mean()
        peak_period_hour = hourly_profile.idxmax().hour
        valley_period_hour = hourly_profile.idxmin().hour
    else:
        peak_period_hour = 0
        valley_period_hour = 0
    
    return {
        'peak_load_MW': peak_val,
        'valley_load_MW': valley_val,
        'mean_load_MW': mean_val,
        'peak_valley_diff_MW': peak_val - valley_val,
        'daily_load_rate': load_rate,
        'peak_duration_hours': peak_hours_count,
        'dr_potential_MW': dr_potential_mw,
        'dr_potential_MWh': dr_potential_mwh,
        'peak_period_hour': peak_period_hour,
        'valley_period_hour': valley_period_hour,
        'shaving_potential_pct': (dr_potential_mw / peak_val * 100) if peak_val > 0 else 0
    }
