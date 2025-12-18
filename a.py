import numpy as np
import pandas as pd
import math
from commonfunc import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def smotth(data, window_size=3, method='moving_average'):
    data_array = np.array(data)
    transposed = data_array.T
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, min(window_size, len(transposed[0]) - 1))

    # 对每一行（原数据的每一列）进行平滑
    if method == 'moving_average':
        smoothed = uniform_filter1d(transposed, size=window_size, axis=1)
    elif method == 'savgol':
        smoothed = savgol_filter(transposed, window_size, 2, axis=1)
    else:
        smoothed = transposed

    # 转置回原始格式
    result = smoothed.T.tolist()
    return result


def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def get_k(S_z, depth):
    # depth = np.array(depth[::-1])
    # S_z = S_z[::-1]
    index = np.where(np.array(depth) > -10)
    S_z = np.array(S_z)[index]
    depth = np.array(depth)[index]

    initial_guess = [10, -0.1, 1]
    # 执行指数拟合
    params, covariance = curve_fit(exponential_func, depth, S_z, p0=initial_guess, maxfev=5000)
    a, b, c = params
    min_depth = -3
    max_depth = -10
    depth_range = np.linspace(min_depth, max_depth, 50)
    count_range = exponential_func(depth_range, a, b, c)
    slop, inrgg = np.polyfit(depth_range, count_range, 1)

    index = np.where(np.array(depth_range) < -9)
    new_dep = np.array(depth_range)[index[0]]
    new_count = np.array(count_range)[index[0]]
    n_slop, n_inrgg = np.polyfit(new_dep, new_count, 1)
    print(n_slop / 2)
    print(slop / 2)

    return n_slop / 2


def get_real_s(wind):
    if wind < 7:
        return 0.0146 * math.sqrt(wind)
    elif wind < 13:
        return 0.003 + 0.00512 * wind
    else:
        return 0.138 * math.log10(wind)


def read_experiment_data(experiment_filepath: str):
    """
    读取实验结果文件
    :param experiment_filepath: 实验结果文件路径
    :return:
    """
    suffix = experiment_filepath.split('.')[-1]
    if suffix == 'csv':
        return pd.read_csv(experiment_filepath)
    elif suffix == 'xlsx':
        return pd.read_excel(experiment_filepath, skiprows=1)
    else:
        raise ValueError('不支持的数据格式')


def get_needData(file: str, flag: bool, out_file):
    original_df = read_experiment_data(file)
    if flag:
        original_df = original_df[original_df['Signal Confidence'] == 4]
    oTime = original_df["Time (sec)"]
    oHeight = original_df["Height (m HAE)"]
    oAlongDistance = original_df["Along-Track (m)"]
    oLatitude = original_df["Latitude (deg)"]
    oLongitude = original_df["Longitude (deg)"]
    new_df = pd.DataFrame(zip(oTime, oAlongDistance, oHeight, oLongitude, oLatitude),
                          columns=['Time', 'AlongDistance', 'Height', 'Longitude', 'Latitude'])
    new_df.to_csv(out_file, mode='w', index=False)
    print("1.获取元数据完成")


def get_waterSurafec(height):
    delta_h = 0.15
    sigma = 0
    min_h = min(height)

    m = int((max(height) - min(height)) / delta_h) + 1
    center_height = []
    seg_count = np.zeros(m, dtype=int)

    for item in height:
        index = int((item - min_h) / delta_h)
        seg_count[index] += 1

    for i in range(1, m + 1):
        ii = min_h + (2 * i - 1) * delta_h / 2
        center_height.append(ii)

    maxpho_index = np.argmax(seg_count)
    maxpho_h = center_height[maxpho_index]
    gauss_height = center_height[maxpho_index - 7:maxpho_index + 8]

    for item in gauss_height:
        sigma = (item - maxpho_h) * (item - maxpho_h)
    water_up = maxpho_h + 2 * sigma
    water_down = maxpho_h - 2 * sigma

    return maxpho_h


def get_chl(file):
    laser_shots = 5714
    n_w = 1.33
    rou_s = 0.0209
    pai = 3.1415926
    T_w = 0.98
    beta_pai = 0.32
    sita = 0.36 * 0.36
    R = 500000

    data_dir = "../yelvsu/1024" + "/"
    # data_dir = "../yelvsu/track-ma1" + "/"
    # data_dir = "2902120_185" + "/"
    # data_dir = "../yelvsu/0327_19_21" + "/"

    distance_path = data_dir + "1selected_distance_along.txt"
    height_path = data_dir + "1selected_height.txt"
    lat_path = data_dir + "1selected_lat.txt"  # 纬度
    lon_path = data_dir + "1selected_lon.txt"
    wind_path = data_dir + "wind.txt"

    alongDis = np.array(read_txt(distance_path))
    height = np.array(read_txt(height_path))
    lat = np.array(read_txt(lat_path))
    lon = np.array(read_txt(lon_path))
    wind = np.array(read_txt(wind_path))

    alongDis = alongDis / 1000

    seg_length = 4
    min_dis = min(alongDis)
    max_dis = max(alongDis)
    seg_count = int((max_dis - min_dis) / seg_length) + 1

    chl_all = [[]]
    depth_all = []
    S_count_all = [[]]
    S_data = [[]]

    lon_seg = []
    lat_seg = []
    wind_seg = []

    lin_al = []
    lin_he = []

    for i in range(seg_count):
        print("------------------------------------", i)
        index_km = np.where((alongDis >= (min_dis + i * seg_length)) & (alongDis < (min_dis + (i + 1) * seg_length)))

        if len(index_km[0]) == 0:
            # last_len = len(S_z_count)
            # new_zero_list = [0] * last_len
            # S_count_all.append(new_zero_list)
            continue

        # 4km / 9km 范围的数据
        index_km = index_km[0]
        dis_km = alongDis[index_km]
        height_refra_km = []
        height_km = height[index_km]
        lon_km = lon[index_km]
        lat_km = lat[index_km]
        wind_km = wind[index_km]

        wind_avg = sum(wind_km) / len(wind_km)
        wind_seg.append(wind_avg)
        lat_seg.append(np.mean(lat_km))
        lon_seg.append(np.mean(lon_km))

        # 沿轨 7m 确实海平面
        max_dis_km = max(dis_km)
        min_dis_km = min(dis_km)
        seg_length_km = 0.007
        seg_count_km = int((max_dis_km - min_dis_km) / seg_length_km) + 1

        for j in range(seg_count_km):
            index_m = np.where(
                (dis_km >= (min_dis_km + j * seg_length_km)) & (dis_km < (min_dis_km + seg_length_km * (j + 1))))

            if len(index_m[0]) == 0:
                continue

            height_m = height_km[index_m[0]]
            distance_m = dis_km[index_m[0]]

            sea_sur = get_waterSurafec(height_m)

            height_refra_m_ = height_m - sea_sur
            height_refra_m = np.array([item * 0.75 if item < 0 else item for item in height_refra_m_])
            height_refra_km.extend(height_refra_m)

            lin_al.extend(distance_m)
            lin_he.extend(height_refra_m)

        height_refra_km = np.array(height_refra_km)

        max_depth = 15
        seg_count_height = int(max_depth / 0.15) + 1

        chl_km = []
        depth_km = []
        S_z = []
        S_z_count = []

        for j in range(seg_count_height):
            h_mid = -j * 0.15
            h_down = h_mid - 0.5
            h_up = h_mid + 0.5
            h_mid = round(h_mid,2)
            index_h = np.where((height_refra_km <= h_up) & (height_refra_km > h_down))

            if len(index_h) == 0 or index_h[0].size == 0:
                N_down = 0
                S_z.append(N_down)
                depth_km.append(h_mid)
                S_z_count.append(N_down)
            else:
                NN = index_h[0].size
                N_down = NN / laser_shots
                S_z.append(N_down)
                depth_km.append(h_mid)
                S_z_count.append(cmath.log(N_down * (R - h_mid) * (R - h_mid)).real)
                # S_z_count.append(cmath.log(N_down * R * R).real)

        #
        # s_df = pd.DataFrame(zip(depth_km, S_z), columns=['lat', 'lon'])
        # out_filepath = "C:/Users\Zoliqin\Desktop/ar_ic/mayue/IO_B/s.csv"
        # s_df.to_csv(out_filepath, index=False)
        depth_all = depth_km
        S_count_all.append(S_z_count)
        S_data.append(S_z)
        print(S_z)

        # print(depth_km[::-1])
        # print(S_z[::-1])

    S_count_all = S_count_all[1:]
    new_s_smo = smotth(S_count_all,5)
    df_count = pd.DataFrame(S_count_all).T
    smoo_df = pd.DataFrame(new_s_smo).T

    S_data = S_data[1:]
    S_data_smo = np.array(smotth(S_data))


    AA = 4 * pai * T_w * T_w * beta_pai / rou_s

    for index in range(len(new_s_smo)):
        print("-------------------------------------------", index)

        if all(item_s == 0 for item_s in new_s_smo[index]):
            continue

        S_z = new_s_smo[index]
        new_alfa = get_k(S_z, depth_all)
        # print(S_data_smo[index])
        # print(depth_all)
        print(new_alfa)
        sita = get_real_s(wind_seg[index])
        print(f"{wind_seg[index]}------{get_real_s(wind_seg[index])}")
        A = AA * sita * new_s_smo[index][0]

        chl_km = []

        for N_down, h_mid in zip(new_s_smo[index], depth_all):
            exp_num = cmath.exp(2 * new_alfa * (-h_mid)).real
            b_bp_z = N_down * exp_num / A
            chl = (b_bp_z / 0.0019) ** (1 / 0.61)

            chl_km.append(b_bp_z.real)
        chl_all.append(chl_km)
        # print(chl_km)


    # # 保存为CSV（无表头，如需表头可添加columns参数）
    # df_count.to_csv("C:/Users\Zoliqin\Desktop/ar_ic/mayue/IO_B/count_min.csv", index=False, header=False)
    # smoo_df.to_csv("C:/Users\Zoliqin\Desktop/ar_ic/mayue/IO_B/count_min_smo.csv", index=False, header=False)

    chl_matrix = chl_all[1:]

    chl_result1 = [[row[i] for row in chl_matrix] for i in range(len(chl_matrix[0]))]
    chl_result2 = np.array(chl_result1)
    # chl_result2 = np.flip(chl_result2, axis=1)
    chl_result3 = smooth2(chl_result2, 2, 2)
    sns.heatmap(chl_result2, vmin=0.001, vmax=0.3, cmap='jet')

    for row in chl_matrix:
        print(row)



    # plt.xticks([0, round((np.size(chl_result2, 1) - 1) / 2), np.size(chl_result2, 1) - 1],
    #            [round(lat[-1],2), (round(lat[0],2) + round(lat[-1],2)) / 2, round(lat[0],2)], rotation=0)
    # plt.yticks([0, round((np.size(chl_result2, 0) - 1) / 2), np.size(chl_result2, 0) - 1], [min(depth_km), 5, max(depth_km)])
    # plt.show()
    #
    plt.xticks([0, round((np.size(chl_result2, 1) - 1) / 2), np.size(chl_result2, 1) - 1],
               [round(lat[0], 2), (round(lat[0], 2) + round(lat[-1], 2)) / 2, round(lat[-1], 2)], rotation=0)
    plt.yticks([0, round((np.size(chl_result2, 0) - 1) / 2), np.size(chl_result2, 0) - 1],
               [max(depth_km), (max(depth_km)+min(depth_km)) / 2, min(depth_km)])
    plt.show()






if __name__ == '__main__':
    out_file1 = "C:/Users\Zoliqin\Desktop/ar_ic/desert/a/1l/chla/data0.csv"
    get_chl(out_file1)
    # get_huadong_chl(out_file1)
