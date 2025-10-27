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

    coordinates_lines = lines[coordinate_start_line: coordinate_start_line + sum(atom_counts)]
    coordinates = [line.split()[:3] for line in coordinates_lines]

    atoms_coordinates = []
    current_index = 0
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atoms_coordinates.append((atom_type, list(map(float, coordinates[current_index]))))
            current_index += 1

    return atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line


def calculate_angle(metal, oxygen1, oxygen2):
    """
    计算金属原子为顶点，两个氧原子为端点的夹角。
    """
    metal = np.array(metal)
    oxygen1 = np.array(oxygen1)
    oxygen2 = np.array(oxygen2)

    vec1 = oxygen1 - metal
    vec2 = oxygen2 - metal

    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("输入向量的模不能为零。")

    dot_product = np.dot(vec1, vec2)
    cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_theta))

    return angle


def process_folder(folder_path):
    """
    处理当前文件夹中的所有.vasp文件。
    """
    # 匹配当前目录下的所有 .vasp 文件
    file_pattern = os.path.join(folder_path, "*.vasp")
    files = glob.glob(file_pattern)

    if not files:
        print("未找到.vasp文件，请检查当前文件夹。")
        return

    results = []

    for poscar_file in files:
        atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_file)

        if len(atom_types) < 2 or len(atoms_coordinates) < 2:
            print(f"文件 {poscar_file} 的原子信息不足，跳过。")
            continue

        M_type = atom_types[0]
        O_type = atom_types[1]

        metal_atoms = [coord for atom, coord in atoms_coordinates if atom == M_type]
        oxygen_atoms = [coord for atom, coord in atoms_coordinates if atom == O_type]

        if len(metal_atoms) == 0 or len(oxygen_atoms) < 2:
            print(f"文件 {poscar_file} 的金属或氧原子数量不足，跳过。")
            continue

        metal = metal_atoms[0]  # 假设只有一个金属原子
        #oxygen1, oxygen2 = oxygen_atoms[:2]  # 任选两个氧原子
        oxygen5 = oxygen_atoms[4]  # 第5个氧原子（索引从0开始）
        oxygen6 = oxygen_atoms[5]  # 第6个氧原子

        angle = calculate_angle(metal, oxygen5, oxygen6)
        results.append(f" {angle:.2f} , 文件: {poscar_file}, 金属: {M_type}, 氧: {O_type}\n")

    result_file = os.path.join(folder_path, "angle_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.writelines(results)

    print(f"角度计算完成，结果已保存到 {result_file}")


if __name__ == "__main__":
    # 使用当前工作目录
    folder_path = os.getcwd()

    if not os.path.exists(folder_path):
        print("文件夹路径不存在，请检查后重新输入。")
    else:
        process_folder(folder_path)
