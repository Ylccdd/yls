import cmath
import numpy as np


def read_txt(file_path):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            data.append(float(line.strip()))
    return data

def compute_k_new(S_h, h, A, k):
    theta_a = 30 #????
    beta_pai = 3.15366 * pow(10, -2)
    a = 0.1
    y1 = (1 + 0.005 * theta_a) * a
    y2 = 4.18 * (1 - 0.52 * cmath.exp(-10.8 * a)) * S_h / (A * cmath.exp(-2 * k * h) * beta_pai)

    y = k - y1 - y2
    return y.real

def compute_k_new_2(S_h, h, A, k):
    theta_a = 30 #????
    beta_pai = 3.15366 * pow(10, -2)
    a = 0.1
    y1 = (1 + 0.005 * theta_a) * a
    y2 = 4.18 * (1 - 0.52 * cmath.exp(-10.8 * a)) * S_h / (0.0287 * cmath.exp(-2 * k * h))

    y = k - y1 - y2
    return y.real
def compute_k(S_h, h, A, k):
    theta_a = 0   #????
    a = 0.1
    y1 = (1 + 0.005 * theta_a) * a
    y2 = 4.18 * (1 - 0.52 * cmath.exp(-10.8 * a)) * S_h / (A * cmath.exp(-2 * k * h))
    y = k - y1 - y2
    return y.real



def smooth2(a, nx, ny):
    from scipy import signal
    kernel = np.ones((nx, ny)) * 1 / (nx * ny)
    r = signal.convolve2d(a, kernel, mode='same')
    return r
