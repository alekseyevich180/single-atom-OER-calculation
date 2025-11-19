import os
import glob
import math
import numpy as np
import sys
current_path = os.path.dirname(os.path.abspath(__file__))  # 当前脚本路径
module_path = r"C:\Users\yingkaiwu\Desktop\single-atom\octachedral_rot\share"  # 根据实际路径调整
sys.path.append(module_path)
import glob
import math
import numpy as np
from atom_location import parse_poscar
from atom_location import calculate_distance
from utils import calculate_plane_normal
from atom_location import calculate_angle
from reorder_atom import reorder_atoms


def rotation_matrix(axis, theta):
    """
    给定旋转轴(axis)和旋转角度theta(弧度)，返回绕该轴旋转theta的旋转矩阵(Rodrigues公式)。
    axis应为归一化后的向量。
    """
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta/2.0)
    b, c, d = -axis * math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = np.array([[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
                    [2*(bc-ad),   aa+cc-bb-dd, 2*(cd-ab)],
                    [2*(bd+ac),   2*(cd+ab),   aa+dd-bb-cc]])
    return rot

if __name__ == "__main__":
    # 提示用户是否执行原子顺序调整程序
    user_input = input("是否执行原子顺序调整程序？(y/n): ").strip().lower()
    if user_input == 'y':
        poscar_file = "Pt.vasp"  # 原始 POSCAR 文件路径
        output_file = "Pt_reordered.vasp"  # 调整顺序后的文件名

        # 定义新的原子顺序（示例：交换第1和第2原子位置）
        new_order = [0, 1, 2, 3, 4, 6, 5]  # 根据原子索引调整顺序

        try:
            reordered_poscar = reorder_atoms(poscar_file, new_order)

            # 保存结果到文件
            with open(output_file, "w") as f:
                f.writelines(reordered_poscar)

            print(f"调整原子顺序后的 POSCAR 文件已保存为: {output_file}")
        except Exception as e:
            print(f"发生错误: {e}")
        
        # 使用重新排序后的文件继续后续操作
        poscar_file = output_file
    else:
        print("跳过原子顺序调整程序，继续使用原始文件 Pt.vasp。")
        poscar_file = "Pt.vasp"

    # 文件处理逻辑开始
    folder_path = "."
    file_pattern = os.path.join(folder_path, poscar_file)
    files = glob.glob(file_pattern)

    if not files:
        print(f"未找到匹配的文件 {poscar_file}，请检查文件名。")
        exit()
    else:
        print(f"找到 {len(files)} 个文件: {files}")

    poscar_file = files[0]
    results = []
    
    # 解析POSCAR文件
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_file)

    # 假设MO6结构：第一个原子类型为M(如Ru、Sn、Ti、Ir等)，第二个原子类型为O
    
    
    M_type = atom_types[0]  # 金属原子类型
    O_type = atom_types[1]  # 氧原子类型
    M_count = atom_counts[0]
    O_count = atom_counts[1] if len(atom_counts) > 1 else 0

    if M_count != 1:
        raise ValueError("未能正确识别单一的中心金属原子，请检查文件。")
    
    if O_count < 5:
        raise ValueError("氧原子数量不足（<5），无法定义平面和目标原子。")
    
    atom_M = int(input("Metal : "))  # 中心金属原子类型
    atom_O1 = int(input("dis_Oxygen : "))
    atom_O6 = int(input("angle_Oxygen1 : "))  # 氧原子类型
    
    M = np.array(atoms_coordinates[atom_M][1])  # 提取金属原子坐标
    atom5 = np.array(atoms_coordinates[atom_O1][1])  # 提取距离计算的氧原子
    O6 = np.array(atoms_coordinates[atom_O6][1])
    # 提取M和O坐标
    
    M = None
    O_list = []
    for atom, coord in atoms_coordinates:
        if atom == M_type and M is None:
            M = np.array(coord)
        elif atom == O_type:
            O_list.append(np.array(coord))

    if M is None:
        raise ValueError(f"未找到{M_type}原子。")
    
    distance = calculate_distance(M, atom5)
    #print(f"1号氧原子与中心{M_type}原子的距离为: {distance}")

    # 使用前4个O定义平面，并选取第5个O为目标旋转原子
    O1, O2, O3, O4 = O_list[0], O_list[1], O_list[2], O_list[3]
    O_target = O_list[4]

    # 定义平面法向量
    scale_factor = float(input("-100 to 100:"))
    normal = calculate_plane_normal(O1, O2, M, scale_factor=scale_factor)

    # O_target相对于M的向量和初始距离
    OM = O_target - M
    initial_distance = np.linalg.norm(OM)

    # 确定要修改的O原子行：假设M在前，O在后
    o_start_line = coordinate_start_line + M_count
    # 第5个O对应行号：o_start_line + 4（下标从0开始）
    target_line_index = o_start_line + 4

    # 请根据实际情况修改final_distance为所需的目标距离
    final_distance = distance

    # 对0到60度，每隔20度一次旋转，并输出文件
    for angle in range(0, 61, 1):
        theta = math.radians(angle)
        R = rotation_matrix(normal, theta)
        OM_rotated = R.dot(OM)
        O_target_new = M + OM_rotated

        magnitude1 = np.linalg.norm(normal)
        magnitude2 = np.linalg.norm(OM_rotated)

        if magnitude1 == 0 or magnitude2 == 0:
            raise ValueError("输入向量的模不能为零。")

        dot_product = np.dot(normal, OM_rotated)
        cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
        vector_angle = math.degrees(math.acos(cos_theta))
    
        surface_angle = 90 - vector_angle


        # 缩放使旋转后O-M距离为目标距离
        current_distance = np.linalg.norm(O_target_new - M)
        scale_factor = distance / current_distance
        O_target_final = M + OM_rotated * scale_factor

        # 修改目标行坐标至缩短后的坐标
        line_split = lines[target_line_index].split()
        line_split[0] = f"{O_target_final[0]:.16f}"
        line_split[1] = f"{O_target_final[1]:.16f}"
        line_split[2] = f"{O_target_final[2]:.16f}"
        new_line = " ".join(line_split) + "\n"
        original_line = lines[target_line_index]
        lines[target_line_index] = new_line

        # 输出文件名带角度标记（使用M_type来生成文件名）
        output_filename = f"{M_type}O6_{angle}.vasp"
        with open(output_filename, "w") as f:
            f.writelines(lines)

        # 恢复原始行以便下个循环使用
        lines[target_line_index] = original_line


        calculated_angle = calculate_angle(M, O6, O_target_final)
        results.append(f"角度: {angle} , 计算的夹角:, {calculated_angle:.2f} , surface_angle:, {surface_angle:.2f}\n")

    with open("calculation_results.txt", "w", encoding="utf-8") as result_file:
        #result_file.write(f"读取的文件: {poscar_file}\n")
        result_file.writelines(results)

    print("已生成0-60度的旋转并缩短距离后的MO6坐标文件。")

