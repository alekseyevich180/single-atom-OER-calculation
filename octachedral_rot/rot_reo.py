import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(BASE_DIR, "share")
if module_path not in sys.path:
    sys.path.append(module_path)

from atom_location import calculate_angle, calculate_distance, parse_poscar
from reorder_atom import reorder_atoms
from utils import calculate_plane_normal

DEFAULT_REORDER_ORDER = [0, 1, 2, 3, 4, 6, 5]


@dataclass
class RotationSettings:
    poscar_pattern: str
    angle_start: int
    angle_end: int
    angle_step: int
    metal_index: Optional[int]
    distance_oxygen_index: int
    angle_oxygen_index: int
    plane_scale: float
    target_o_offset: int
    reorder: bool
    reorder_output: Optional[str]
    reorder_sequence: Optional[List[int]]
    bundle_outputs: bool


def rotation_matrix(axis, theta):
    """Rodrigues 旋转矩阵。"""
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd - ab)],
            [2 * (bd + ac), 2 * (cd + ab), aa + dd - bb - cc],
        ]
    )
    return rot


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 par/命令行参数控制 Ir-O6 旋转。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--par", help="par 文件路径（key=value），默认自动查找")
    parser.add_argument("--poscar", help="POSCAR 文件或通配符，覆盖 par 中的 POSCAR_FILE")
    parser.add_argument("--angle-start", type=int, help="起始角度（度）")
    parser.add_argument("--angle-end", type=int, help="结束角度（度，包含）")
    parser.add_argument("--angle-step", type=int, help="角度步长（度）")
    parser.add_argument(
        "--metal-index", type=int, help="POSCAR 中中心金属原子索引（0 基，默认自动）"
    )
    parser.add_argument(
        "--distance-oxygen-index",
        type=int,
        help="用于保持 M-O 距离的氧原子索引（0 基）",
    )
    parser.add_argument(
        "--angle-oxygen-index",
        type=int,
        help="用于计算 M-O-O 夹角的氧原子索引（0 基）",
    )
    parser.add_argument(
        "--plane-scale",
        type=float,
        help="calculate_plane_normal 的 scale_factor（-100 ~ 100）",
    )
    parser.add_argument(
        "--target-o-offset",
        type=int,
        help="在氧原子列表（按类型顺序）中要旋转的氧编号（0 基）",
    )
    parser.add_argument(
        "--reorder",
        dest="reorder",
        action="store_true",
        help="强制执行 reorder_atoms 步骤",
    )
    parser.add_argument(
        "--no-reorder",
        dest="reorder",
        action="store_false",
        help="强制跳过 reorder_atoms 步骤",
    )
    parser.set_defaults(reorder=None)
    parser.add_argument(
        "--reorder-output",
        help="reorder 后的输出 POSCAR，默认在原文件名后添加 _reordered",
    )
    parser.add_argument(
        "--reorder-seq",
        help="逗号分隔的新索引顺序，如 0,1,2,3,4,6,5，用于 reorder_atoms",
    )
    parser.add_argument(
        "--bundle",
        dest="bundle",
        action="store_true",
        help="将生成的结构与结果文件集中存放到 <planeScale>_<angleEnd>/ 目录下",
    )
    parser.add_argument(
        "--no-bundle",
        dest="bundle",
        action="store_false",
        help="禁用批量输出目录",
    )
    parser.set_defaults(reorder=None, bundle=None)
    return parser.parse_args()


def autodetect_par(search_dir: str) -> Optional[str]:
    env_path = os.environ.get("PAR_FILE")
    candidates = [
        env_path,
        os.path.join(search_dir, "par"),
        os.path.join(search_dir, "par_all"),
        os.path.join(os.path.dirname(search_dir), "par"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def load_par(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.isfile(path):
        return {}
    cfg: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip().upper()] = value.strip()
    return cfg


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "t", "yes", "y", "on"}


def format_scale_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{value}"
    if "e" in text or "E" in text:
        return text.replace("E", "e")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"非法的 reorder 顺序: {value}") from exc


def pick_numeric(
    arg_value: Optional[float],
    cfg: Dict[str, str],
    key: str,
    *,
    cast,
    default: Optional[float] = None,
) -> Optional[float]:
    if arg_value is not None:
        return arg_value
    raw = cfg.get(key.upper())
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{key} 的值无法转换: {raw}") from exc


def build_settings(args) -> RotationSettings:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    par_path = args.par or autodetect_par(script_dir)
    cfg = load_par(par_path)

    poscar_pattern = (
        args.poscar or cfg.get("POSCAR_FILE") or cfg.get("POSCAR") or "Ir.vasp"
    )
    angle_start = int(pick_numeric(args.angle_start, cfg, "ANGLE_START", cast=int, default=0))
    angle_end = int(
        pick_numeric(args.angle_end, cfg, "ANGLE_END", cast=int, default=90)
    )
    angle_step = int(
        pick_numeric(args.angle_step, cfg, "ANGLE_STEP", cast=int, default=1)
    )
    if angle_step <= 0:
        raise ValueError("ANGLE_STEP 必须为正数。")
    if angle_end < angle_start:
        raise ValueError("ANGLE_END 需要大于或等于 ANGLE_START。")

    metal_index_val = pick_numeric(
        args.metal_index, cfg, "METAL_INDEX", cast=int, default=None
    )
    metal_index = int(metal_index_val) if metal_index_val is not None else None

    distance_idx = pick_numeric(
        args.distance_oxygen_index,
        cfg,
        "DISTANCE_OXYGEN_INDEX",
        cast=int,
        default=None,
    )
    angle_idx = pick_numeric(
        args.angle_oxygen_index,
        cfg,
        "ANGLE_OXYGEN_INDEX",
        cast=int,
        default=None,
    )
    plane_scale_val = pick_numeric(
        args.plane_scale, cfg, "PLANE_SCALE", cast=float, default=0.0
    )
    if distance_idx is None or angle_idx is None:
        missing = []
        if distance_idx is None:
            missing.append("DISTANCE_OXYGEN_INDEX")
        if angle_idx is None:
            missing.append("ANGLE_OXYGEN_INDEX")
        raise ValueError(f"缺少必要参数: {', '.join(missing)}。")

    target_o_offset_val = pick_numeric(
        args.target_o_offset, cfg, "TARGET_O_OFFSET", cast=int, default=4
    )
    target_o_offset = int(target_o_offset_val)
    if target_o_offset < 0:
        raise ValueError("TARGET_O_OFFSET 必须 >= 0。")

    if args.reorder is not None:
        reorder_flag = args.reorder
    else:
        reorder_flag = parse_bool(cfg.get("REORDER"), default=False)

    reorder_sequence = args.reorder_seq or cfg.get("REORDER_SEQ") or cfg.get("REORDER_SEQUENCE")
    reorder_sequence_list = parse_int_list(reorder_sequence)
    reorder_output = args.reorder_output or cfg.get("REORDER_OUTPUT")
    if args.bundle is not None:
        bundle_flag = args.bundle
    else:
        bundle_flag = parse_bool(cfg.get("BUNDLE_OUTPUT"), default=True)

    return RotationSettings(
        poscar_pattern=poscar_pattern,
        angle_start=angle_start,
        angle_end=angle_end,
        angle_step=angle_step,
        metal_index=metal_index,
        distance_oxygen_index=int(distance_idx),
        angle_oxygen_index=int(angle_idx),
        plane_scale=float(plane_scale_val),
        target_o_offset=target_o_offset,
        reorder=reorder_flag,
        reorder_output=reorder_output,
        reorder_sequence=reorder_sequence_list,
        bundle_outputs=bundle_flag,
    )


def prepare_poscar(settings: RotationSettings) -> str:
    pattern = settings.poscar_pattern
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到匹配 POSCAR: {pattern}")
    poscar_file = files[0]
    if not settings.reorder:
        return poscar_file

    sequence = settings.reorder_sequence or DEFAULT_REORDER_ORDER
    output = (
        settings.reorder_output
        or f"{os.path.splitext(poscar_file)[0]}_reordered.vasp"
    )
    reordered_lines = reorder_atoms(poscar_file, sequence)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(reordered_lines)
    print(f"已根据 {sequence} 对原子排序，输出 {output}")
    return output


def run_rotation(settings: RotationSettings):
    poscar_file = prepare_poscar(settings)
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(
        poscar_file
    )

    if len(atom_types) < 2:
        raise ValueError("POSCAR 中需要包含至少两种元素（M 和 O）。")

    M_type = atom_types[0]
    O_type = atom_types[1]
    M_count = atom_counts[0]
    O_count = atom_counts[1] if len(atom_counts) > 1 else 0

    if M_count != 1:
        raise ValueError("未能识别唯一的中心金属原子，请检查 POSCAR。")
    if O_count < settings.target_o_offset + 1:
        raise ValueError("氧原子数量不足，无法选择目标氧。")

    if settings.metal_index is not None:
        if not (0 <= settings.metal_index < len(atoms_coordinates)):
            raise ValueError("METAL_INDEX 超出原子数范围。")
        M = np.array(atoms_coordinates[settings.metal_index][1])
    else:
        M = None
        for atom, coord in atoms_coordinates:
            if atom == M_type:
                M = np.array(coord)
                break
        if M is None:
            raise ValueError(f"未找到 {M_type} 原子。")

    for idx in (settings.distance_oxygen_index, settings.angle_oxygen_index):
        if not (0 <= idx < len(atoms_coordinates)):
            raise ValueError("氧原子索引超出范围。")

    atom_distance = np.array(atoms_coordinates[settings.distance_oxygen_index][1])
    angle_reference = np.array(atoms_coordinates[settings.angle_oxygen_index][1])

    O_list = [
        np.array(coord) for atom, coord in atoms_coordinates if atom == O_type
    ]
    if len(O_list) <= settings.target_o_offset + 1:
        raise ValueError("氧原子数量不足以定义平面和目标原子。")

    O1, O2, O3, O4 = O_list[0], O_list[1], O_list[2], O_list[3]
    O_target = O_list[settings.target_o_offset]

    distance = calculate_distance(M, atom_distance)
    normal = calculate_plane_normal(O1, O2, M, scale_factor=settings.plane_scale)

    OM = O_target - M

    o_start_line = coordinate_start_line + M_count
    target_line_index = o_start_line + settings.target_o_offset

    if not (0 <= target_line_index < len(lines)):
        raise ValueError("计算得到的氧坐标行超出范围。")

    bundle_dir = None
    if settings.bundle_outputs:
        scale_label = format_scale_label(settings.plane_scale)
        bundle_name = f"{scale_label}_{settings.angle_end}"
        bundle_dir = os.path.abspath(bundle_name)
        os.makedirs(bundle_dir, exist_ok=True)

    results = []
    for angle in range(settings.angle_start, settings.angle_end + 1, settings.angle_step):
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

        current_distance = np.linalg.norm(O_target_new - M)
        distance_scale = distance / current_distance
        O_target_final = M + OM_rotated * distance_scale

        original_line = lines[target_line_index]
        line_split = original_line.split()
        line_split[0] = f"{O_target_final[0]:.16f}"
        line_split[1] = f"{O_target_final[1]:.16f}"
        line_split[2] = f"{O_target_final[2]:.16f}"
        new_line = " ".join(line_split) + "\n"
        lines[target_line_index] = new_line

        output_filename = f"{M_type}O6_{angle}.vasp"
        output_path = (
            os.path.join(bundle_dir, output_filename)
            if bundle_dir
            else os.path.abspath(output_filename)
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        lines[target_line_index] = original_line

        calculated_angle = calculate_angle(M, angle_reference, O_target_final)
        results.append(
            f"角度， {angle} , 计算的夹角， {calculated_angle:.2f} , surface_angle， {surface_angle:.2f}\n"
        )

    results_path = (
        os.path.join(bundle_dir, "calculation_results.txt")
        if bundle_dir
        else os.path.abspath("calculation_results.txt")
    )
    with open(results_path, "w", encoding="utf-8") as result_file:
        result_file.writelines(results)
    print(
        f"已完成{settings.angle_start}° 到 {settings.angle_end}° 的旋转扫描，结果写入 {results_path}。"
    )


def main():
    args = parse_args()
    settings = build_settings(args)
    run_rotation(settings)


if __name__ == "__main__":
    main()
