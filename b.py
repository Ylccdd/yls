import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# -------------------------- 论文固定参数配置 --------------------------
# ICESat-2 系统参数
H = 500000  # 卫星轨道高度 (m)
laser_wavelength = 532  # 激光波长 (nm)
beta_pi = 0.32  # 散射角π处的后向散射相函数 (sr⁻¹)
t = 0.98  # 气-水界面透过率
k = 1.0  # 后向散射-衰减对数比
gamma = 0.78  # bbp波长转换幂律斜率
Kd_correction = 1.4  # Kd天顶角差异校正因子

# 数据分箱参数
along_track_bin = 4000  # 沿轨分箱大小 (m)
depth_bin = 1  # 深度分箱大小 (m)
depth_step = 0.15  # 深度滑动步长 (m)
max_depth = 15  # 最大有效探测深度 (m)
slope_depth_range = [3, 10]  # 斜率法计算深度范围 (m)

# -------------------------- 核心工具函数 --------------------------
def read_atl03_data(atl03_path):
    """读取ICESat-2 ATL03数据（TXT格式），提取预处理数据"""
    # 读取TXT文件的辅助函数
    def read_txt_file(file_path):
        values = []
        with open(file_path, 'r') as f:
            for line in f:
                # 文件格式：每行一个数值
                value = line.strip()
                if value:
                    values.append(float(value))
        return np.array(values)
    
    # 数据文件路径
    base_path = r'd:\workspace\PycharmProjects\a\a\yelvsu\yelvsu\1007'
    
    # 读取各数据文件
    lon = read_txt_file(base_path + '\\1selected_lon.txt')
    lat = read_txt_file(base_path + '\\1selected_lat.txt')
    height = read_txt_file(base_path + '\\1selected_height.txt')
    along_track = read_txt_file(base_path + '\\1selected_distance_along.txt')
    
    # 返回DataFrame
    return pd.DataFrame({
        'lat': lat, 'lon': lon, 'height': height, 'along_track': along_track
    })

def get_sea_surface(df):
    """确定海平面高程（直方图峰值法）"""
    delta_h = 0.1
    heights = df['height'].values
    min_h, max_h = heights.min(), heights.max()
    bins = np.arange(min_h, max_h + delta_h, delta_h)
    counts, _ = np.histogram(heights, bins=bins)
    max_count_idx = np.argmax(counts)
    sea_surface = bins[max_count_idx] + delta_h/2  # 峰值bin中心为海平面
    return sea_surface

def preprocess_data(df):
    """数据预处理：海平面校正、折射校正、滑动累积"""
    # 1. 确定海平面并计算水下深度（折射校正：实际深度 = 测量深度 / 1.33）
    sea_surface = get_sea_surface(df)
    df['depth'] = (sea_surface - df['height']) / 1.33  # 水下深度（正值向下）
    df = df[df['depth'] >= 0]  # 仅保留水下光子
    
    # 2. 沿轨分箱（4km）
    min_along = df['along_track'].min()
    max_along = df['along_track'].max()
    along_bins = np.arange(min_along, max_along + along_track_bin, along_track_bin)
    
    # 3. 深度分箱（1m）+ 滑动累积（0.15m步长）
    depth_bins = np.arange(0, max_depth + depth_bin, depth_step)
    signal_matrix = []  # 存储每个沿轨bin的深度-信号
    along_centers = []  # 沿轨bin中心坐标
    
    for i in range(len(along_bins)-1):
        # 提取当前沿轨bin数据
        bin_mask = (df['along_track'] >= along_bins[i]) & (df['along_track'] < along_bins[i+1])
        bin_df = df[bin_mask]
        if len(bin_df) < 100:  # 光子数过少则跳过
            continue
        
        # 深度方向滑动累积
        bin_signal = []
        for d in depth_bins[:-1]:
            depth_mask = (bin_df['depth'] >= d) & (bin_df['depth'] < d + depth_bin)
            photon_count = depth_mask.sum()
            bin_signal.append(photon_count / 10000)  # 归一化（激光重复频率10kHz）
        
        signal_matrix.append(bin_signal)
        along_centers.append((along_bins[i] + along_bins[i+1])/2000)  # 转换为km
    
    return np.array(signal_matrix).T, depth_bins[:-1], np.array(along_centers)

def afterpulse_deconvolution(Sm):
    """后脉冲去卷积（论文公式3-6）"""
    n_depth, n_along = Sm.shape
    S = np.zeros_like(Sm)  # 去卷积后的真实信号
    
    for i in range(n_along):
        # 1. 找到当前沿轨bin的信号峰值位置p
        signal_i = Sm[:, i]
        p = np.argmax(signal_i)
        if p < 1 or p + 10 >= n_depth:  # 确保有足够的邻域bin
            S[:, i] = signal_i
            continue
        
        # 2. 计算脉冲响应函数F(z)（12个邻域bin：p-1到p+10）
        neighbor_signals = signal_i[p-1:p+11]
        F = neighbor_signals / neighbor_signals.sum()
        F_matrix = np.zeros((n_depth, n_depth))
        
        # 3. 构建F(z)矩阵（下三角托普利兹矩阵）
        for j in range(n_depth):
            for k in range(j+1):
                if j - k < len(F):
                    F_matrix[j, k] = F[j - k]
        
        # 4. 反卷积求解真实信号S = F⁻¹ * Sm（添加正则化避免病态）
        F_inv = np.linalg.inv(F_matrix + 1e-6 * np.eye(n_depth))
        S[:, i] = F_inv @ signal_i
    
    return S

def slope_method(S, depth):
    """斜率法计算参考边界衰减系数α(zm)（论文公式7-9）"""
    # 筛选3-10m深度范围
    depth_mask = (depth >= slope_depth_range[0]) & (depth <= slope_depth_range[1])
    z = depth[depth_mask]
    S_z = S[depth_mask]
    
    # 范围校正信号：RCS(z) = S(z) * (H + z)²
    RCS = S_z * (H + z) ** 2
    log_RCS = np.log(RCS + 1e-8)  # 避免log(0)
    
    # 线性拟合：log_RCS = a - 2αz → 斜率 = -2α
    coeffs = np.polyfit(z, log_RCS, 1)
    slope = coeffs[0]
    alpha_zm = -slope / 2  # 参考边界衰减系数
    zm = z[-1]  # 参考边界深度（10m）
    return alpha_zm, zm

def klett_method(S, depth, alpha_zm, zm):
    """Klett方法反演全深度Kd（论文公式10-13）"""
    n_depth = len(depth)
    Kd = np.zeros(n_depth)
    RCS = S * (H + depth) ** 2  # 范围校正信号
    
    # 计算积分项 ∫RCS(z)dz
    integral_RCS = np.zeros(n_depth)
    for i in range(n_depth):
        integral_RCS[i] = trapezoid(RCS[:i+1], depth[:i+1])
    
    # 分深度段计算α(z)（Kd≈α(z)）
    zm_idx = np.argmin(np.abs(depth - zm))
    RCS_zm = RCS[zm_idx]
    integral_zm = integral_RCS[zm_idx]
    
    for i in range(n_depth):
        ratio = (RCS[i] / RCS_zm) ** (1/k)
        integral_i = integral_RCS[i]
        integral_term = (integral_i ** (1/k) - integral_zm ** (1/k)) / RCS_zm ** (1/k)
        
        if depth[i] <= zm:
            denominator = 1/alpha_zm + (2/k) * integral_term
        else:
            denominator = 1/alpha_zm - (2/k) * integral_term
        
        Kd[i] = ratio / denominator if denominator != 0 else alpha_zm
    
    return Kd * Kd_correction  # 应用天顶角校正因子

def retrieve_Kd(S_deconv, depth):
    """Kd反演：斜率法+Klett方法"""
    n_along = S_deconv.shape[1]
    Kd_matrix = []
    
    for i in range(n_along):
        alpha_zm, zm = slope_method(S_deconv[:, i], depth)
        Kd = klett_method(S_deconv[:, i], depth, alpha_zm, zm)
        Kd_matrix.append(Kd)
    
    return np.array(Kd_matrix).T

def retrieve_bbp(S_deconv, Kd_matrix, depth, wind_data):
    """bbp反演（论文公式14-17）"""
    n_depth, n_along = S_deconv.shape
    bbp_matrix = []
    theta = 0  # 激光天顶角（假设为0°）
    
    for i in range(n_along):
        # 1. 计算波斜率方差σ²（由风速）
        wind = wind_data[i] if i < len(wind_data) else 5  # 风速默认值
        if wind < 7:
            sigma2 = 0.0146 * np.sqrt(wind)
        elif 7 <= wind < 13:
            sigma2 = 0.003 + 0.00512 * wind
        else:
            sigma2 = 0.138 * np.log10(wind)
        
        # 2. 海面理论后向散射βs
        cos_theta = np.cos(np.radians(theta))
        beta_s = 0.0209 / (4 * np.pi * sigma2 * cos_theta**4) * np.exp(-(np.tan(np.radians(theta)))**2 / (2*sigma2))
        
        # 3. 校准常数A
        S0 = S_deconv[0, i]  # 海面处信号（深度=0）
        RCS0 = S0 * H**2  # 海面RCS
        log_RCS0 = np.log(RCS0 + 1e-8)
        A = log_RCS0 / beta_s
        
        # 4. 计算bbp(z)
        bbp = []
        Kd = Kd_matrix[:, i]
        for j in range(n_depth):
            exp_term = np.exp(2 * Kd[j] * depth[j])
            RCS_z = S_deconv[j, i] * (H + depth[j])**2
            log_RCSz = np.log(RCS_z + 1e-8)
            bbp_z = (log_RCSz * exp_term) / (A * beta_pi * t**2)
            bbp.append(np.abs(bbp_z))  # 确保非负
        
        bbp_matrix.append(bbp)
    
    return np.array(bbp_matrix).T

def wavelength_conversion(data, src_wave, target_wave=532, param_type='Kd'):
    """波长转换（论文公式1-2）"""
    if param_type == 'Kd':
        # Kd(532) = 0.68*(Kd(490)-0.022) + 0.054
        if src_wave == 490:
            return 0.68 * (data - 0.022) + 0.054
        else:
            raise ValueError("Kd仅支持490nm转532nm")
    elif param_type == 'bbp':
        # bbp(532) = bbp(λ) * (λ/532)^γ
        return data * (src_wave / target_wave) ** gamma

def validate_results(retrieved, observed, param_name):
    """验证：计算R²和RMSE"""
    retrieved_flat = retrieved.flatten()
    observed_flat = observed.flatten()
    valid_mask = (retrieved_flat > 0) & (observed_flat > 0)
    
    if valid_mask.sum() < 10:
        return {'R²': 0, 'RMSE': 0}
    
    r2 = np.corrcoef(retrieved_flat[valid_mask], observed_flat[valid_mask])[0,1]**2
    rmse = np.sqrt(np.mean((retrieved_flat[valid_mask] - observed_flat[valid_mask])**2))
    return {'R²': round(r2, 3), 'RMSE': round(rmse, 4)}

# -------------------------- 主流程执行 --------------------------
if __name__ == '__main__':
    # 1. 数据输入（替换为实际文件路径）
    atl03_path = ""  # ICESat-2 ATL03数据路径（不再使用）
    modis_path = "MODIS_monthly_Kd_bbp.csv"  # MODIS月均数据（含Kd490、bbp443）
    wind_path = r'd:\workspace\PycharmProjects\a\a\yelvsu\yelvsu\1007\wind.txt'  # 风速数据（TXT格式）
    
    # 2. 数据读取与预处理
    print("步骤1：读取并预处理ICESat-2数据...")
    atl03_df = read_atl03_data(atl03_path)
    Sm, depth, along_km = preprocess_data(atl03_df)  # Sm：测量信号矩阵（深度×沿轨）
    
    # 3. 后脉冲去卷积
    print("步骤2：后脉冲去卷积...")
    S = afterpulse_deconvolution(Sm)  # S：真实信号矩阵
    
    # 4. 反演Kd和bbp
    print("步骤3：反演漫衰减系数Kd...")
    Kd_matrix = retrieve_Kd(S, depth)
    
    print("步骤4：反演颗粒后向散射系数bbp...")
    # 读取风速数据（TXT格式：每行一个数值）
    def read_wind_txt(file_path):
        values = []
        with open(file_path, 'r') as f:
            for line in f:
                value = line.strip()
                if value:
                    values.append(float(value))
        return np.array(values)
    wind_data = read_wind_txt(wind_path)  # 沿轨风速
    bbp_matrix = retrieve_bbp(S, Kd_matrix, depth, wind_data)
    
    # 5. 结果可视化（论文图2、3风格）
    print("步骤5：结果可视化...")
    
    # 5.1 生成热力图（深度-沿轨分布）并直接显示
    print("显示热力图（深度-沿轨分布）...")
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 10))
    
    # Kd垂直-水平分布热力图
    sns.heatmap(Kd_matrix, ax=axes1[0], cmap='jet', vmin=0, vmax=0.4,
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes1[0].set_title('Kd Distribution', fontsize=12)
    axes1[0].set_xlabel('Along-Track Distance (km)')
    axes1[0].set_ylabel('Depth (m)')
    
    # bbp垂直-水平分布热力图（对数尺度）
    sns.heatmap(np.log10(bbp_matrix + 1e-6), ax=axes1[1], cmap='jet', vmin=-3, vmax=-1,
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes1[1].set_title('bbp Distribution (log10)', fontsize=12)
    axes1[1].set_xlabel('Along-Track Distance (km)')
    axes1[1].set_ylabel('Depth (m)')
    
    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示热力图
    
    # 5.2 生成时间序列对比图（ICESat2）并直接显示
    print("显示时间序列图（沿轨分布）...")
    # 尝试使用不同的数据处理方式，避免无效值问题
    try:
        # 使用表面层数据（深度较浅的部分）来计算沿轨平均
        surface_depth_idx = np.where(depth <= 5)[0]  # 使用5m以内的表层数据
        
        if len(surface_depth_idx) > 0:
            # 提取表层数据
            surface_Kd = Kd_matrix[surface_depth_idx, :]
            surface_bbp = bbp_matrix[surface_depth_idx, :]
            
            # 沿轨平均（只计算有效值）
            avg_Kd = np.zeros(surface_Kd.shape[1])
            avg_bbp = np.zeros(surface_bbp.shape[1])
            
            for i in range(surface_Kd.shape[1]):
                # 计算每个沿轨位置的有效平均值
                valid_Kd = surface_Kd[:, i][np.isfinite(surface_Kd[:, i])]
                valid_bbp = surface_bbp[:, i][np.isfinite(surface_bbp[:, i])]
                
                avg_Kd[i] = np.mean(valid_Kd) if len(valid_Kd) > 0 else np.nan
                avg_bbp[i] = np.mean(valid_bbp) if len(valid_bbp) > 0 else np.nan
            
            # 过滤掉无效值
            valid_idx = np.isfinite(avg_Kd) & np.isfinite(avg_bbp) & (avg_bbp > 0)
            
            if np.sum(valid_idx) > 5:  # 确保有足够的数据点
                fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
                
                # Kd时间序列图
                axes2[0].plot(along_km[valid_idx], avg_Kd[valid_idx], 'o-', color='blue', label='ICESat2')
                axes2[0].set_title('Kd Along-Track Distribution', fontsize=12)
                axes2[0].set_xlabel('Along-Track Distance (km)')
                axes2[0].set_ylabel('Kd (m⁻¹)')
                axes2[0].set_ylim(0, 0.4)
                axes2[0].legend()
                
                # bbp时间序列图（对数尺度，使用较小的范围）
                axes2[1].plot(along_km[valid_idx], avg_bbp[valid_idx], 'o-', color='blue', label='ICESat2')
                axes2[1].set_title('bbp Along-Track Distribution', fontsize=12)
                axes2[1].set_xlabel('Along-Track Distance (km)')
                axes2[1].set_ylabel('bbp (m⁻¹)')
                axes2[1].set_ylim(1e-3, 1e-1)  # 设置合理的y轴范围
                axes2[1].set_yscale('log')
                axes2[1].legend()
                
                plt.tight_layout()
                plt.show()  # 显示时间序列图，阻塞等待用户关闭
            else:
                print("有效数据点不足，无法生成时间序列图")
        else:
            print("没有找到足够的表层数据")
    except Exception as e:
        print(f"生成时间序列图时出错: {e}")
        # 如果出错，使用简化版本的时间序列图
        try:
            print("尝试使用简化版本的时间序列图...")
            fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
            
            # 只使用第一个深度层的数据
            if Kd_matrix.shape[0] > 0:
                # Kd时间序列图（使用第一个深度层）
                kd_first_depth = Kd_matrix[0, :]
                valid_kd_idx = np.isfinite(kd_first_depth)
                if np.sum(valid_kd_idx) > 0:
                    axes2[0].plot(along_km[valid_kd_idx], kd_first_depth[valid_kd_idx], 'o-', color='blue', label='ICESat2')
                axes2[0].set_title('Kd Along-Track Distribution (First Depth Layer)', fontsize=12)
                axes2[0].set_xlabel('Along-Track Distance (km)')
                axes2[0].set_ylabel('Kd (m⁻¹)')
                axes2[0].set_ylim(0, 0.4)
                axes2[0].legend()
            
            # bbp时间序列图（使用第一个深度层）
            if bbp_matrix.shape[0] > 0:
                bbp_first_depth = bbp_matrix[0, :]
                valid_bbp_idx = np.isfinite(bbp_first_depth) & (bbp_first_depth > 0)
                if np.sum(valid_bbp_idx) > 0:
                    axes2[1].plot(along_km[valid_bbp_idx], bbp_first_depth[valid_bbp_idx], 'o-', color='blue', label='ICESat2')
                axes2[1].set_title('bbp Along-Track Distribution (First Depth Layer)', fontsize=12)
                axes2[1].set_xlabel('Along-Track Distance (km)')
                axes2[1].set_ylabel('bbp (m⁻¹)')
                axes2[1].set_ylim(1e-3, 1e-1)
                axes2[1].set_yscale('log')
                axes2[1].legend()
            
            plt.tight_layout()
            plt.show()
        except Exception as e2:
            print(f"简化版本时间序列图也生成失败: {e2}")