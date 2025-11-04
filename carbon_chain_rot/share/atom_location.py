import os
import glob
import math
import re
import argparse
import numpy as np

def _safe_float(token: str) -> float:
    """Parse floats that may include CIF-style uncertainty like 0.123(4)."""
    m = re.match(r"^[\+\-]?\d*(?:\.\d*)?(?:[eE][\+\-]?\d+)?", token.strip())
    if m and m.group(0):
        return float(m.group(0))
    return float(token)


def _lattice_from_abc(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Build lattice vectors (3x3) from a,b,c and alpha,beta,gamma (degrees)."""
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ax, ay, az = a, 0.0, 0.0
    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0
    cx = c * math.cos(beta)
    # Guard against division by zero when gamma ~ 0 or 180
    if abs(math.sin(gamma)) < 1e-12:
        cy = 0.0
    else:
        cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    # Ensure numerical stability for cz
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    cz = math.sqrt(cz_sq)
    return [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]


def _frac_to_cart(frac, lattice):
    ax, ay, az = lattice[0]
    bx, by, bz = lattice[1]
    cx, cy, cz = lattice[2]
    fx, fy, fz = frac
    x = fx * ax + fy * bx + fz * cx
    y = fx * ay + fy * by + fz * cy
    z = fx * az + fy * bz + fz * cz
    return [x, y, z]


def parse_poscar(poscar_path):
    """
    解析POSCAR/CONTCAR（VASP5/常见布局），返回原子类型与其坐标。

    注意：
    - 支持识别 Direct/Cartesian，并在 Direct 时转换为笛卡尔坐标（Å）。
    - 解析晶格（缩放 + 3 行基矢）。
    - 若缺少元素类型行（VASP4 风格），类型将用 'X' 占位。

    返回（为兼容现有调用，保持原签名）：
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line
    """
    with open(poscar_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    # 去除尾部换行，保留原始以兼容返回
    lines = [ln.rstrip("\n\r") for ln in raw_lines]
    if len(lines) < 8:
        raise ValueError("POSCAR 文件行数不足，无法解析")

    # 缩放因子
    try:
        scale = float(lines[1].split()[0])
    except Exception as e:
        raise ValueError(f"无法解析缩放因子: {e}")
    scale = abs(scale)

    # 晶格基矢
    try:
        a_vec = [float(x) for x in lines[2].split()[:3]]
        b_vec = [float(x) for x in lines[3].split()[:3]]
        c_vec = [float(x) for x in lines[4].split()[:3]]
    except Exception as e:
        raise ValueError(f"无法解析晶格基矢: {e}")
    lattice = [[v * scale for v in a_vec],
               [v * scale for v in b_vec],
               [v * scale for v in c_vec]]

    # 解析元素与数目（VASP5/4 兼容）
    line5 = lines[5].split()
    line6 = lines[6].split() if len(lines) > 6 else []
    if line5 and all(tok.isdigit() for tok in line5):
        # VASP4：第6行为计数
        atom_types = []
        atom_counts = [int(x) for x in line5]
        coord_type_line_idx = 7
    else:
        atom_types = line5
        atom_counts = [int(x) for x in line6]
        coord_type_line_idx = 7

    if len(lines) <= coord_type_line_idx:
        raise ValueError("POSCAR 缺少坐标类型行")

    selective = False
    coord_type_line = lines[coord_type_line_idx].strip().lower()
    if coord_type_line.startswith('s'):
        selective = True
        coord_type_line_idx += 1
        coord_type_line = lines[coord_type_line_idx].strip().lower()

    direct = True
    if coord_type_line.startswith('d'):
        direct = True
    elif coord_type_line.startswith('c') or coord_type_line.startswith('k'):
        # 一些文件使用 'cartesian' 或 'k' 表示笛卡尔
        direct = False
    else:
        # 默认按 Direct 处理
        direct = True

    coordinate_start_line = coord_type_line_idx + 1
    natoms = sum(atom_counts)
    coord_lines = lines[coordinate_start_line: coordinate_start_line + natoms]
    if len(coord_lines) < natoms:
        raise ValueError("POSCAR 坐标行数量不足")

    coords = []
    for ln in coord_lines:
        parts = ln.split()
        if len(parts) < 3:
            raise ValueError("POSCAR 坐标行格式错误")
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        if direct:
            cart = _frac_to_cart([x, y, z], lattice)
        else:
            cart = [x, y, z]
        coords.append(cart)

    # 若未提供元素类型（极少见），用占位符
    if not atom_types or len(atom_types) != len(atom_counts):
        atom_types = ["X"] * len(atom_counts)

    atoms_coordinates = []
    idx = 0
    for typ, cnt in zip(atom_types, atom_counts):
        for _ in range(cnt):
            atoms_coordinates.append((typ, coords[idx]))
            idx += 1

    return atoms_coordinates, raw_lines, atom_types, atom_counts, coordinate_start_line

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


def parse_cif(cif_path):
    """
    解析 CIF 文件：读取晶胞 a,b,c,alpha,beta,gamma 与分数坐标，转换为笛卡尔坐标。

    返回（仿照 parse_poscar 的前五项签名，以便轻松过渡）：
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line
    其中 coordinate_start_line 对于 CIF 无语义，返回 -1。
    """
    with open(cif_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()
    lines = [ln.rstrip('\r\n') for ln in raw_lines]

    # 读取晶胞参数
    def find_tag_value(tag_names):
        for ln in lines:
            if not ln.strip() or ln.strip().startswith('#'):
                continue
            for tag in tag_names:
                if ln.lower().startswith(tag.lower() + ' '):
                    parts = ln.split()
                    if len(parts) >= 2:
                        return _safe_float(parts[1])
        return None

    a = find_tag_value(["_cell_length_a"]) or 0.0
    b = find_tag_value(["_cell_length_b"]) or 0.0
    c = find_tag_value(["_cell_length_c"]) or 0.0
    alpha = find_tag_value(["_cell_angle_alpha"]) or 90.0
    beta = find_tag_value(["_cell_angle_beta"]) or 90.0
    gamma = find_tag_value(["_cell_angle_gamma"]) or 90.0
    lattice = _lattice_from_abc(a, b, c, alpha, beta, gamma)

    # 查找包含分数坐标的 loop_
    headers = []
    data_rows = []
    in_loop = False
    capture_rows = False
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if s.lower().startswith('loop_'):
            in_loop = True
            headers = []
            data_rows = []
            capture_rows = False
            continue
        if in_loop and s.startswith('_'):
            headers.append(s)
            continue
        if in_loop and headers:
            # 第一行非下划线行，视为数据起始
            capture_rows = True
        if capture_rows:
            # loop 结束条件：遇到新 loop_ 或新 tag
            if s.lower().startswith('loop_') or s.startswith('_'):
                # 处理结束，推回一行（简单实现：关闭采集，后续逻辑忽略多 loop）
                in_loop = False
                capture_rows = False
                # 不继续采集更多 loop，以保持简单；可扩展为支持多个 loop
                break
            else:
                data_rows.append(s)

    # 解析包含分数坐标的 loop（寻找 fract_x/y/z）
    lower_headers = [h.lower() for h in headers]
    try:
        ix = lower_headers.index('_atom_site_fract_x')
        iy = lower_headers.index('_atom_site_fract_y')
        iz = lower_headers.index('_atom_site_fract_z')
    except ValueError:
        raise ValueError('CIF 中未找到 _atom_site_fract_{x,y,z} 列')

    symbol_idx = None
    for key in ['_atom_site_type_symbol', '_atom_site_label']:
        if key in lower_headers:
            symbol_idx = lower_headers.index(key)
            break
    if symbol_idx is None:
        # 若缺少类型列，使用通用占位符 X
        symbol_idx = -1

    atoms_coordinates = []
    species_order = []
    species_counts = {}
    for row in data_rows:
        parts = row.split()
        if max(ix, iy, iz) >= len(parts):
            continue
        fx = _safe_float(parts[ix])
        fy = _safe_float(parts[iy])
        fz = _safe_float(parts[iz])
        sym = 'X'
        if 0 <= symbol_idx < len(parts):
            sym = parts[symbol_idx]
        cart = _frac_to_cart([fx, fy, fz], lattice)
        atoms_coordinates.append((sym, cart))
        if sym not in species_counts:
            species_counts[sym] = 0
            species_order.append(sym)
        species_counts[sym] += 1

    atom_types = species_order
    atom_counts = [species_counts[s] for s in species_order]
    return atoms_coordinates, raw_lines, atom_types, atom_counts, -1


def read_structure(path):
    """
    统一读取结构文件：支持 .cif 与 VASP POSCAR/CONTCAR/vasp。

    返回：
    - atoms_coordinates: [(symbol, [x,y,z])...] 笛卡尔坐标（Å）
    - meta: dict，包含 {format, atom_types, atom_counts}
    """
    ext = os.path.splitext(path)[1].lower()
    fmt = None
    if ext in ['.cif']:
        atoms_coordinates, _, atom_types, atom_counts, _ = parse_cif(path)
        fmt = 'cif'
    else:
        atoms_coordinates, _, atom_types, atom_counts, _ = parse_poscar(path)
        fmt = 'poscar'
    meta = {
        'format': fmt,
        'atom_types': atom_types,
        'atom_counts': atom_counts,
    }
    return atoms_coordinates, meta


def _print_atoms(atoms):
    lines = []
    for idx, (sym, coord) in enumerate(atoms):
        x, y, z = coord
        lines.append(f"{idx:4d}  {sym:>2s}   {x: .6f}  {y: .6f}  {z: .6f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Read POSCAR/CIF, list atoms, compute distances and angles."
    )
    parser.add_argument("path", help="Path to structure file (.cif or POSCAR/CONTCAR)")
    parser.add_argument("--format", choices=["auto", "poscar", "cif"], default="auto",
                        help="Force input format (default: auto by extension)")
    parser.add_argument("--list", action="store_true", help="List atoms with indices")
    parser.add_argument("--distance", nargs=2, type=int,
                        metavar=("I", "J"), help="Distance between atom I and J")
    parser.add_argument("--angle", nargs=3, type=int,
                        metavar=("I", "J", "K"), help="Angle I-J-K in degrees")
    parser.add_argument("--out", default=None, help="Write output to file instead of stdout")

    args = parser.parse_args()

    path = args.path
    if not os.path.isfile(path):
        raise SystemExit(f"Input file not found: {path}")

    # Choose reader
    ext = os.path.splitext(path)[1].lower()
    if args.format == "cif":
        atoms_coordinates, _, atom_types, atom_counts, _ = parse_cif(path)
        meta = {"format": "cif", "atom_types": atom_types, "atom_counts": atom_counts}
        atoms = atoms_coordinates
    elif args.format == "poscar":
        atoms_coordinates, _, atom_types, atom_counts, _ = parse_poscar(path)
        meta = {"format": "poscar", "atom_types": atom_types, "atom_counts": atom_counts}
        atoms = atoms_coordinates
    else:
        atoms, meta = read_structure(path)

    out_lines = []

    if args.list or (not args.distance and not args.angle):
        out_lines.append("# Atoms (index  symbol    x        y        z) [Angstrom]")
        out_lines.append(_print_atoms(atoms))

    # helpers to get coords by index
    coords = [coord for (_sym, coord) in atoms]

    if args.distance:
        i, j = args.distance
        n = len(coords)
        if not (0 <= i < n and 0 <= j < n):
            raise SystemExit(f"Distance indices out of range: 0..{n-1}")
        d = calculate_distance(coords[i], coords[j])
        out_lines.append(f"\n# Distance: {i}-{j} = {d:.6f} Å")

    if args.angle:
        i, j, k = args.angle
        n = len(coords)
        if not (0 <= i < n and 0 <= j < n and 0 <= k < n):
            raise SystemExit(f"Angle indices out of range: 0..{n-1}")
        ang = calculate_angle(coords[j], coords[i], coords[k])
        out_lines.append(f"\n# Angle: {i}-{j}-{k} = {ang:.6f} deg")

    text = "\n".join(out_lines) + ("\n" if out_lines else "")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
