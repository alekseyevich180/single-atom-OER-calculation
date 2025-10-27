import numpy as np
import math

def calculate_plane_normal(point1, point2, point3, scale_factor=1.0):
    """
    根据三点定义平面，计算平面法向量。
    
    参数:
        - point1, point2, point3: numpy 数组，表示平面上的三点坐标。
        - scale_factor: float，用于调整点之间的权重，默认值为 1.0。
    
    返回:
        - normal: numpy 数组，归一化的平面法向量。
    """
    # 按照 scale_factor 调整点的权重
    vector1 = point2 - scale_factor * point1
    vector2 = point3 - point1

    # 计算法向量并归一化
    normal = np.cross(vector1, vector2)
    normal = normal / np.linalg.norm(normal)  # 归一化
  
    return normal

def vector_angle(point1, point2, point3, scale_factor=1.0):
    """
    计算两个向量之间的夹角。

    参数:
        - vector1, vector2: numpy 数组，表示两个向量。

    返回:
        - angle: float，两个向量之间的夹角（单位：度）。
    """
    
    vector1 = point2 - scale_factor * point1
    vector2 = point3 - point1
    
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("输入向量的模不能为零。")

    dot_product = np.dot(vector1, vector2)
    cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_theta))

    return angle
