import os
import glob
import math
import numpy as np

def parse_poscar(poscar_path):
    """
    解析POSCAR文件，将每个原子类型与其坐标对应起来。
    """
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    coordinate_start_line = 8

    if lines[7].strip().lower() in ["selective dynamics", "s"]:
        coordinate_start_line += 1

    coordinates_lines = lines[coordinate_start_line : coordinate_start_line + sum(atom_counts)]
    coordinates = [line.split()[:3] for line in coordinates_lines]

    atoms_coordinates = []
    current_index = 0
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atoms_coordinates.append((atom_type, list(map(float, coordinates[current_index]))))
            current_index += 1

    return atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line

def calculate_distance(atom1, atom2):
    """
    计算两个原子之间的距离。

    参数:
        - atom1: tuple，包含原子类型和坐标 (atom_type, [x, y, z])。
        - atom2: tuple，包含原子类型和坐标 (atom_type, [x, y, z])。

    返回:
        - distance: float，两个原子之间的距离。
    """
    coord1 = atom1
    coord2 = atom2
    distance = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)))
    
    return distance    


def calculate_angle(metal, oxygen1, oxygen2):
    """
    计算金属原子为顶点，两个氧原子为端点的夹角。
    """
    # 确保输入是 numpy 数组
    metal = np.array(metal)
    oxygen1 = np.array(oxygen1)
    oxygen2 = np.array(oxygen2)

    # 计算向量
    vec1 = oxygen1 - metal
    vec2 = oxygen2 - metal

    # 调试打印向量
    #print(f"向量 vec1: {vec1}")
    #print(f"向量 vec2: {vec2}")

    # 检查向量长度
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    #print(f"模长 vec1: {magnitude1}, vec2: {magnitude2}")
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("输入向量的模不能为零。")

    # 计算点积
    dot_product = np.dot(vec1, vec2)
    #print(f"点积: {dot_product}")

    # 计算夹角
    cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_theta))
    #print(f"角度: {angle} 度")

    return angle