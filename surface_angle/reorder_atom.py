from atom_location import parse_poscar  # 假设 parse_poscar 在 poscar_utils.py 中

def reorder_atoms(poscar_path, new_order):
    """
    调用 parse_poscar 解析 POSCAR 文件，并根据 new_order 调整原子顺序。
    返回调整顺序后的 POSCAR 内容。
    
    参数：
        - poscar_path: str，POSCAR 文件路径
        - new_order: list，指定新的原子顺序（索引从 0 开始）
    
    返回：
        - new_lines: list，调整顺序后的 POSCAR 内容
    """
    # 调用 parse_poscar 解析文件
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_path)

    # 按 new_order 重新排序
    reordered_atoms = [atoms_coordinates[i] for i in new_order]

    # 更新原子数据信息
    reordered_coordinates = [f"{x[1][0]:.16f} {x[1][1]:.16f} {x[1][2]:.16f}" for x in reordered_atoms]

    # 更新 lines 中的坐标部分
    total_atoms = sum(atom_counts)
    coordinate_lines = lines[coordinate_start_line : coordinate_start_line + total_atoms]
    for i, line in enumerate(reordered_coordinates):
        coordinate_lines[i] = line + "\n"

    # 替换回 lines 中
    new_lines = lines[:coordinate_start_line] + coordinate_lines + lines[coordinate_start_line + total_atoms:]

    return new_lines

# 主程序调用
if __name__ == "__main__":
    poscar_file = "IrO6.vasp"  # 原始 POSCAR 文件路径
    output_file = "IrO6_reordered.vasp"  # 调整顺序后的文件名

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
