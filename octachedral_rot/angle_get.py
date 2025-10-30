import os
import glob
import math
import numpy as np
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
module_path = r"C:\Users\yingkaiwu\Desktop\single-atom\octachedral_rot\share"
sys.path.append(module_path)
from atom_location import parse_poscar
from atom_location import calculate_angle
from utils import calculate_plane_normal

def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd - ab)],
                    [2 * (bd + ac), 2 * (cd + ab), aa + dd - bb - cc]])
    return rot

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\\d+)', s)]

if __name__ == "__main__":
    folder_path = "."
    file_pattern = os.path.join(folder_path, "*.vasp")
    files = sorted(glob.glob(file_pattern), key=lambda x: natural_sort_key(os.path.basename(x)))

    if not files:
        print("未找到匹配的.vasp文件。")
        exit()
    else:
        print(f"找到 {len(files)} 个文件: {files}")

    results = []

    for poscar_file in files:
        print(f"正在处理文件: {poscar_file}")

        # 解析POSCAR文件
        atoms_coordinates, _, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_file)

        if len(atom_types) < 2 or len(atom_counts) < 2:
            print(f"文件 {poscar_file} 中的原子类型或计数不足，跳过。")
            continue

        M_type = atom_types[0]  # 金属原子类型
        O_type = atom_types[1]  # 氧原子类型
        M_count = atom_counts[0]
        O_count = atom_counts[1]

        if M_count != 1:
            print(f"文件 {poscar_file} 中未能识别单一的中心金属原子，跳过。")
            continue

        if O_count < 5:
            print(f"文件 {poscar_file} 中氧原子数量不足（<5），跳过。")
            continue

        M = None
        O_list = []
        for atom, coord in atoms_coordinates:
            if atom == M_type and M is None:
                M = np.array(coord)
            elif atom == O_type:
                O_list.append(np.array(coord))

        if M is None or len(O_list) < 6:
            print(f"文件 {poscar_file} 中无法找到所需的原子坐标，跳过。")
            continue

        # 定义用于计算的原子
        O1, O2, O3, O4, O5, O6 = O_list[:6]

        # 计算平面法向量
        normal = calculate_plane_normal(O1, O2, M)

        # 计算夹角
        OM = O5 - M
        magnitude1 = np.linalg.norm(normal)
        magnitude2 = np.linalg.norm(OM)

        if magnitude1 == 0 or magnitude2 == 0:
            print(f"文件 {poscar_file} 中向量的模为零，跳过。")
            continue

        dot_product = np.dot(normal, OM)
        cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
        vector_angle = math.degrees(math.acos(cos_theta))

        # 计算表面角度
        surface_angle = 90 - vector_angle

        calculated_angle = calculate_angle(M, O6, O5)

        results.append(f"文件: {os.path.basename(poscar_file)}, 夹角: , {calculated_angle:.2f}, 表面角度:, {surface_angle:.2f}\n")

    results.sort(key=lambda x: natural_sort_key(x.split(",")[0].split(":")[1].strip()))

    with open("calculation_results.txt", "w", encoding="utf-8") as result_file:
        result_file.writelines(results)

    print("已完成对所有.vasp文件的夹角测量，并输出结果至 calculation_results.txt。")
