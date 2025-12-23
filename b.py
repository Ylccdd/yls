import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.linalg import lstsq

# -------------------------- 论文固定参数配置 --------------------------
# ICESat-2 系统参数
H = 500000  # 卫星轨道高度 (m)
laser_wavelength = 532  # 激光波长 (nm)
beta_pi = 0.32  # 散射角π处的后向散射相函数 (sr⁻¹)
cont_t = 0.98  # 气-水界面透过率
k = 1.0  # 后向散射-衰减对数比
gamma = 0.78  # bbp波长转换幂律斜率
Kd_correction = 1.4  # Kd天顶角差异校正因子

# Chl反演参数（论文公式18，532nm波长依赖参数）
phi_lambda = 0.0019  # φ(532)，参考论文参考文献[26,48]
psi_lambda = 0.61    # ψ(532)，参考论文参考文献[26,48]

# bbp不确定性分析参数（论文5.2节）
A_uncertainty = 0.2  # 校准常数A的相对不确定性（±20%）
beta_pi_uncertainty = 0.1  # β(π)的相对不确定性（±10%）

# 数据分箱参数
along_track_bin = 4000  # 沿轨分箱大小 (m)
depth_bin = 1  # 深度分箱大小 (m)
depth_step = 0.15  # 深度滑动步长 (m)
max_depth = 15  # 最大有效探测深度 (m)
slope_depth_range = [3, 10]  # 斜率法计算深度范围 (m)

# -------------------------- 核心工具函数 --------------------------
def read_atl03_data(csv_path):
    """读取ICESat-2 ATL03数据（CSV格式），提取预处理数据"""
    # 从CSV文件读取数据，跳过第一行，使用第二行作为列名
    df = pd.read_csv(csv_path, skiprows=1)
    
    # 重命名列以匹配原有代码结构
    df = df.rename(columns={
        'along_distance': 'along_track',
        'conf': 'confidence'
    })
    
    # 转换为数值类型
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['along_track'] = pd.to_numeric(df['along_track'], errors='coerce')
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    
    # 返回所需列
    return df[['lat', 'lon', 'height', 'along_track', 'confidence']]

def get_sea_surface(df):
    """确定海平面高程（直方图峰值法）"""
    delta_h = 0.1
    
    # 由于已经在预处理第一步筛选了置信度=4的光子，这里直接使用全部数据
    heights = df['height'].values
    
    # 去除NaN值
    heights = heights[np.isfinite(heights)]
    
    # 确保有足够的数据
    if len(heights) < 10:
        return 25.0  # 默认海平面高度
    
    min_h, max_h = heights.min(), heights.max()
    
    # 确保min_h < max_h
    if min_h >= max_h:
        return min_h  # 如果所有高度相同，返回该高度作为海平面
    
    bins = np.arange(min_h, max_h + delta_h, delta_h)
    counts, _ = np.histogram(heights, bins=bins)
    max_count_idx = np.argmax(counts)
    sea_surface = bins[max_count_idx] + delta_h/2  # 峰值bin中心为海平面
    return sea_surface

def preprocess_data(df):
    """数据预处理：海平面校正、折射校正、滑动累积"""
    # 1. 筛选置信度=4的光子
    df = df[df['confidence'] == 4]
    
    # 2. 过滤掉height为负值的项
    df = df[df['height'] >= 0]
    
    # 3. 确定海平面并计算水下深度（折射校正：实际深度 = 测量深度 / 1.33）
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

def afterpulse_deconvolution(Sm, reg_lambda=1e-4, depth=None, output_csv='test/deconvolution_records.csv'):
    """后脉冲去卷积（增强稳定性：Tikhonov正则化替代直接求逆）（论文公式3-6）"""
    import csv
    
    n_depth, n_along = Sm.shape
    S = np.zeros_like(Sm)  # 去卷积后的真实信号
    
    # 直接赋值去卷积结果
    S = Sm

    # # 初始化CSV文件
    # with open(output_csv, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     # 写入表头
    #     writer.writerow(['segment_id', 'depth', 'f_before', 'p_bin_idx', 
    #                      'p_minus_1', 'p', 'p_plus_1', 'p_plus_2', 'p_plus_3', 
    #                      'p_plus_4', 'p_plus_5', 'p_plus_6', 'p_plus_7', 
    #                      'p_plus_8', 'p_plus_9', 'p_plus_10', 'f_after'])
    
    # for i in range(n_along):
    #     # 1. 找到当前沿轨bin的信号峰值位置p
    #     signal_i = Sm[:, i]
    #     p = np.argmax(signal_i)
        
    #     if p == 0
    #         p += 1

    #     # 2. 直接记录当前沿轨段的所有深度bin数据（初始状态）
    #     temp_S = signal_i.copy()  # 临时存储去卷积结果
        
    #     # 3. 记录当前沿轨段的所有深度bin数据
    #     for j in range(n_depth):
    #         # 计算当前bin的深度值
    #         current_depth = depth[j] if depth is not None else j
            
    #         # 处理前的f值
    #         f_before = Sm[j, i]
            
    #         # 进行去卷积处理
    #         if p >= 1 and p + 10 < n_depth:  # 确保有足够的邻域bin
                
    #             # 峰值常量
    #             peak_s = signal_i[p-1 : p+11]  
    #             sum_peak = sum(peak_s)
                
    #             # 构建F(z)矩阵
    #             F = signal_i[j-1 : j+11]
                
    #             F_matrix = np.zeros((12, 12))
                
    #             # 每行循环构建
    #             for z in range(12):
    #                 for k in range(12):
    #                     idx = z + 1 - k
    #                     if 0 <= idx < len(F):
    #                         F_matrix[z, k] = F[idx]

    #             F = F / sum_peak  # 归一化得到脉冲响应函数
                
    #             # 2.3 增强稳定性：Tikhonov正则化求解
    #             F_T = F_matrix.T
    #             regularization = reg_lambda * np.eye(n_depth)
    #             F_T_F = F_T @ F_matrix + regularization
    #             F_T_Sm = F_T @ signal_i
    #             temp_S = lstsq(F_T_F, F_T_Sm, overwrite_a=True, overwrite_b=True)[0]
            
    #             # 3. 准备p-1到p+10的值
    #             # 初始化数组，确保有12个元素
    #             p_values = np.zeros(12)
                
    #             # 填充p-1到p+10的值，处理边界情况
    #             for idx_offset in range(-1, 11):
    #                 pos = p + idx_offset
    #                 if 0 <= pos < n_depth:
    #                     p_values[idx_offset + 1] = signal_i[pos]
    #                 # 否则保持为0

    #         # 去卷积之后的值
    #         f_after = temp_S[j]
            
    #         # 写入CSV文件
    #         with open(output_csv, 'a', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow([i, current_depth, f_before, p, 
    #                              p_values[0], p_values[1], p_values[2], p_values[3], p_values[4], 
    #                              p_values[5], p_values[6], p_values[7], p_values[8], 
    #                              p_values[9], p_values[10], p_values[11], f_after])
        
    #     # 保存去卷积结果
    #     S[:, i] = temp_S
    
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
    print(f"zm: {zm:.2f}, α(zm): {alpha_zm:.4f}")
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
    print(f"zm_idx: {zm_idx}, RCS(zm): {RCS_zm:.4f}, ∫RCS(zm): {integral_zm:.4f}")
    
    # 处理RCS_zm为0的情况
    if RCS_zm <= 1e-10:
        RCS_zm = 1e-10
    
    for i in range(n_depth):
        # 处理RCS[i]为0的情况
        if RCS[i] <= 1e-10:
            Kd[i] = alpha_zm
            continue
        
        ratio = (RCS[i] / RCS_zm) ** (1/k)
        integral_i = integral_RCS[i]
        
        # 处理积分项为0的情况
        if integral_i <= 1e-10 or integral_zm <= 1e-10:
            Kd[i] = alpha_zm
            continue
            
        try:
            integral_term = abs( (integral_i ** (1/k) - integral_zm ** (1/k)) ) / RCS_zm ** (1/k)
        except:
            Kd[i] = alpha_zm
            continue
        
        if depth[i] <= zm:
            denominator = 1/alpha_zm + (2/k) * integral_term
        else:
            denominator = 1/alpha_zm - (2/k) * integral_term
        
        # 使用近似0的范围判断，避免浮点数精度问题
        epsilon = 1e-10  # 极小值阈值
        Kd[i] = ratio / denominator if abs(denominator) > epsilon else alpha_zm
    
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
        if wind < 0:
            wind = 5  # 确保风速为正值
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
        
        # 处理beta_s为0的情况
        if abs(beta_s) < 1e-10:
            beta_s = 1e-10
        
        A = log_RCS0 / beta_s
        
        # 处理A为0或无穷大的情况
        if abs(A) < 1e-10 or not np.isfinite(A):
            A = 1.0
        
        # 4. 计算bbp(z)
        bbp = []
        Kd = Kd_matrix[:, i]
        for j in range(n_depth):
            # 处理Kd为NaN或无穷大的情况
            if not np.isfinite(Kd[j]) or Kd[j] < 0:
                Kd_j = 0.05  # 使用默认值
            else:
                Kd_j = Kd[j]
            
            exp_term = np.exp(2 * Kd_j * depth[j])
            RCS_z = S_deconv[j, i] * (H + depth[j])**2
            log_RCSz = np.log(RCS_z + 1e-8)
            bbp_z = (log_RCSz * exp_term) / (A * beta_pi * cont_t**2)
            bbp_value = np.abs(bbp_z)  # 确保非负
            
            # 处理bbp_value为NaN或无穷大的情况
            if not np.isfinite(bbp_value) or bbp_value > 10:
                bbp_value = 1e-3  # 使用默认值
            
            bbp.append(bbp_value)
        
        bbp_matrix.append(bbp)
    
    return np.array(bbp_matrix).T



def calculate_chl(bbp_matrix):
    """Chl反演（论文公式18：bbp(z)=φ(λ)×Chl(z)^ψ(λ)）"""
    # 公式变形：Chl(z) = (bbp(z) / φ(λ))^(1/ψ(λ))
    chl_matrix = (bbp_matrix / phi_lambda) ** (1 / psi_lambda)
    # 限制Chl合理范围（论文中Track1峰值24.62 mg/m³，Track3峰值2.46 mg/m³）
    chl_matrix = np.clip(chl_matrix, 0.1, 30)
    return chl_matrix

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
    # 1. 数据输入（使用CSV文件路径）
    csv_path = r'd:\workspace\PycharmProjects\a\a\yelvsu\yelvsu\1007\0_20201007_0209_gt1l_27.5-31.5_step1.csv'  # ICESat-2 ATL03数据路径（CSV格式）
    modis_path = "MODIS_monthly_Kd_bbp.csv"  # MODIS月均数据（含Kd490、bbp443）
    wind_path = r'd:\workspace\PycharmProjects\a\a\yelvsu\yelvsu\1007\wind.txt'  # 风速数据（TXT格式）
    
    # 2. 数据读取与预处理
    print("步骤1：读取并预处理ICESat-2数据...")
    atl03_df = read_atl03_data(csv_path)
    Sm, depth, along_km = preprocess_data(atl03_df)  # Sm：测量信号矩阵（深度×沿轨）
    
    # 3. 后脉冲去卷积（增强稳定性）
    print("步骤2：后脉冲去卷积（Tikhonov正则化）...")
    S = afterpulse_deconvolution(Sm, reg_lambda=1e-5, depth=depth)  # S：真实信号矩阵
    
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
    
    # 6. 补充模块2：Chl反演
    print("步骤6：叶绿素a浓度（Chl）反演...")
    chl_matrix = calculate_chl(bbp_matrix)
    print(f"Chl浓度范围：{chl_matrix.min():.2f} ~ {chl_matrix.max():.2f} mg/m³（论文Track1峰值：24.62 mg/m³）")
    
    # 7. 结果可视化
    print("步骤7：结果可视化...")
    
    # 确保输出目录存在
    import os
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 7.0 去卷积前后信号热力图对比
    print("步骤7.0：绘制去卷积前后信号热力图...")
    fig0, axes0 = plt.subplots(2, 1, figsize=(12, 16))
    
    # 去卷积前原始信号热力图
    sns.heatmap(Sm, ax=axes0[0], cmap='jet', vmin=0, vmax=np.percentile(Sm, 95),
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes0[0].set_title('Deconvolution Before - Raw Signal Matrix (depth×along-track)', fontsize=12)
    axes0[0].set_xlabel('Along-Track Distance (km)')
    axes0[0].set_ylabel('Depth (m)')
    
    # 去卷积后信号热力图
    sns.heatmap(S, ax=axes0[1], cmap='jet', vmin=0, vmax=np.percentile(S[np.isfinite(S)], 95),
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes0[1].set_title('Deconvolution After - Corrected Signal Matrix (depth×along-track)', fontsize=12)
    axes0[1].set_xlabel('Along-Track Distance (km)')
    axes0[1].set_ylabel('Depth (m)')
    
    plt.tight_layout()
    # 保存为PNG文件
    plt.savefig(os.path.join(output_dir, '0_deconvolution_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    # 7.1 核心参数热力图（Kd/bbp/Chl）
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 15))
    
    # Kd分布（纵轴范围更长）
    sns.heatmap(Kd_matrix, ax=axes1[0], cmap='jet', vmin=0, vmax=0.8,  # 增大vmax从0.4到0.8，纵轴更长
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes1[0].set_title('Kd Distribution (m⁻¹)', fontsize=12)
    axes1[0].set_xlabel('Along-Track Distance (km)')
    axes1[0].set_ylabel('Depth (m)')
    
    # bbp分布（对数尺度）
    sns.heatmap(np.log10(bbp_matrix + 1e-6), ax=axes1[1], cmap='jet', vmin=-3, vmax=0,
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes1[1].set_title('bbp Distribution (log10, m⁻¹)', fontsize=12)
    axes1[1].set_xlabel('Along-Track Distance (km)')
    axes1[1].set_ylabel('Depth (m)')
    
    # Chl分布（线性尺度，最大值40）
    sns.heatmap(chl_matrix, ax=axes1[2], cmap='jet', vmin=0, vmax=40,
                xticklabels=np.round(along_km, 1), yticklabels=np.round(depth, 1))
    axes1[2].set_title('Chlorophyll-a Distribution (mg/m³)', fontsize=12)
    axes1[2].set_xlabel('Along-Track Distance (km)')
    axes1[2].set_ylabel('Depth (m)')
    
    plt.tight_layout()
    # 保存为PNG文件
    plt.savefig(os.path.join(output_dir, '1_core_parameters_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    # 7.3 沿轨分布对比图（Kd/bbp/Chl）
    try:
        # 表层数据沿轨平均（5m以内）
        surface_idx = np.where(depth <= 5)[0]
        if len(surface_idx) > 0:
            avg_Kd = np.nanmean(Kd_matrix[surface_idx, :], axis=0)
            avg_bbp = np.nanmean(bbp_matrix[surface_idx, :], axis=0)
            avg_Chl = np.nanmean(chl_matrix[surface_idx, :], axis=0)
            
            valid_idx = np.isfinite(avg_Kd) & np.isfinite(avg_bbp) & np.isfinite(avg_Chl)
            
            fig3, axes3 = plt.subplots(3, 1, figsize=(12, 12))
            # Kd沿轨分布（纵轴范围更长）
            axes3[0].plot(along_km[valid_idx], avg_Kd[valid_idx], 'o-', color='blue', label='ICESat2')
            axes3[0].set_title('Kd Along-Track Distribution', fontsize=12)
            axes3[0].set_xlabel('Along-Track Distance (km)')
            axes3[0].set_ylabel('Kd (m⁻¹)')
            axes3[0].set_ylim(0, 0.8)  # 增大纵轴范围，从0-0.4改为0-0.8
            axes3[0].legend()
            
            # bbp沿轨分布（对数尺度）
            axes3[1].plot(along_km[valid_idx], avg_bbp[valid_idx], 'o-', color='green', label='ICESat2')
            axes3[1].set_title('bbp Along-Track Distribution', fontsize=12)
            axes3[1].set_xlabel('Along-Track Distance (km)')
            axes3[1].set_ylabel('bbp (m⁻¹)')
            axes3[1].set_yscale('log')
            axes3[1].set_ylim(1e-3, 1e0)  # 扩大对数尺度范围（从0.0001到1.0）
            axes3[1].legend()
            
            # Chl沿轨分布
            axes3[2].plot(along_km[valid_idx], avg_Chl[valid_idx], 'o-', color='red', label='ICESat2')
            axes3[2].set_title('Chlorophyll-a Along-Track Distribution', fontsize=12)
            axes3[2].set_xlabel('Along-Track Distance (km)')
            axes3[2].set_ylabel('Chl (mg/m³)')
            axes3[2].set_ylim(0, 40)
            axes3[2].legend()
            
            plt.tight_layout()
            # 保存为PNG文件
            plt.savefig(os.path.join(output_dir, '2_along_track_distribution.png'), dpi=300, bbox_inches='tight')
            plt.show()
    except Exception as e:
        print(f"沿轨分布图画图出错：{e}")
    
    print("所有模块执行完成！")