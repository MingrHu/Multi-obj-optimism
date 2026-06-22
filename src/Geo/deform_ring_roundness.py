#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 DEFORM 3D .KEY 文件中读取指定对象的 RZ 节点坐标和 ELMCON 单元连接，
提取与 z = 常数平面的截面边界，并计算内、外轮廓的圆度。

默认对象名称：Workpiece - Ring
默认截面：z = 0
圆度默认采用最小区域圆（Minimum Zone Circle, MZC）：
    圆度 = R_max - R_min
其中 R_max、R_min 是同心包容圆的最大、最小半径。

依赖：
    numpy
    matplotlib
    scipy   （用于最小区域圆优化；若缺失，仍会输出最小二乘圆结果）

示例：
    python deform_ring_roundness.py model.key
    python deform_ring_roundness.py model.key --plane-z 0 --object-name "Workpiece - Ring"
    python deform_ring_roundness.py model.key --object-id 1 --samples 4000
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


HEX8_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


@dataclass
class Mesh:
    object_id: int
    object_name: str
    node_ids: np.ndarray          # (N,)
    coordinates: np.ndarray       # (N, 3)
    elements: np.ndarray          # (M, 8)


@dataclass
class CircleResult:
    method: str
    center_x: float
    center_y: float
    radius: float
    roundness: float
    r_min: float
    r_max: float


@dataclass
class ProfileResult:
    name: str
    raw_vertex_count: int
    sample_count: int
    perimeter: float
    polygon_area: float
    least_squares: CircleResult
    minimum_zone: CircleResult | None


def parse_float(text: str) -> float:
    """兼容 Fortran 的 D 指数写法。"""
    return float(text.replace("D", "E").replace("d", "e"))


def first_token(line: str) -> str:
    parts = line.strip().split()
    return parts[0].upper() if parts else ""


def scan_object_names(lines: Sequence[str]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if parts and parts[0].upper() == "OBJNAM" and len(parts) >= 2:
            try:
                object_id = int(parts[1])
            except ValueError:
                i += 1
                continue
            name = lines[i + 1].strip() if i + 1 < len(lines) else ""
            names[object_id] = name
            i += 2
        else:
            i += 1
    return names


def choose_object_id(
    names: Dict[int, str],
    object_id: int | None,
    object_name: str | None,
) -> int:
    if object_id is not None:
        if names and object_id not in names:
            known = ", ".join(f"{k}: {v!r}" for k, v in sorted(names.items()))
            raise ValueError(f"未找到对象编号 {object_id}。现有对象：{known}")
        return object_id

    target = (object_name or "Workpiece - Ring").strip().casefold()

    exact = [oid for oid, name in names.items() if name.strip().casefold() == target]
    if len(exact) == 1:
        return exact[0]

    partial = [oid for oid, name in names.items() if target in name.strip().casefold()]
    if len(partial) == 1:
        return partial[0]

    known = ", ".join(f"{k}: {v!r}" for k, v in sorted(names.items()))
    if not names:
        raise ValueError("KEY 文件中未找到 OBJNAM 字段，请使用 --object-id 指定对象编号。")
    if len(exact) > 1 or len(partial) > 1:
        raise ValueError(f"对象名称匹配到多个对象，请使用 --object-id。现有对象：{known}")
    raise ValueError(f"未找到对象名称 {object_name!r}。现有对象：{known}")


def _next_data_line(lines: Sequence[str], index: int) -> Tuple[int, List[str]]:
    """跳过空行和以 * 开头的注释行，返回下一条数据行。"""
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("*"):
            return index, stripped.split()
        index += 1
    raise EOFError("读取数据块时文件提前结束。")


def read_rz_block(
    lines: Sequence[str],
    start: int,
    count: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    node_ids = np.empty(count, dtype=np.int64)
    xyz = np.empty((count, 3), dtype=np.float64)

    i = start
    for row in range(count):
        i, parts = _next_data_line(lines, i)
        if len(parts) < 4:
            raise ValueError(f"RZ 第 {row + 1} 条记录字段不足：{lines[i].rstrip()}")
        try:
            node_ids[row] = int(parts[0])
            xyz[row, 0] = parse_float(parts[1])
            xyz[row, 1] = parse_float(parts[2])
            xyz[row, 2] = parse_float(parts[3])
        except ValueError as exc:
            raise ValueError(f"RZ 第 {row + 1} 条记录无法解析：{lines[i].rstrip()}") from exc
        i += 1

    return node_ids, xyz, i


def read_elmcon_block(
    lines: Sequence[str],
    start: int,
    count: int,
    nodes_per_element: int,
) -> Tuple[np.ndarray, int]:
    if nodes_per_element != 8:
        raise ValueError(
            f"当前程序只实现了 8 节点六面体单元，文件中 ELMCON 的节点数为 {nodes_per_element}。"
        )

    elements = np.empty((count, nodes_per_element), dtype=np.int64)
    i = start

    for row in range(count):
        i, parts = _next_data_line(lines, i)
        expected = 1 + nodes_per_element
        if len(parts) < expected:
            raise ValueError(
                f"ELMCON 第 {row + 1} 条记录应至少有 {expected} 个字段，"
                f"实际为 {len(parts)}：{lines[i].rstrip()}"
            )
        try:
            # parts[0] 是单元编号，后续 8 个整数是节点编号。
            elements[row, :] = [int(v) for v in parts[1:expected]]
        except ValueError as exc:
            raise ValueError(
                f"ELMCON 第 {row + 1} 条记录无法解析：{lines[i].rstrip()}"
            ) from exc
        i += 1

    return elements, i


def read_deform_key(
    key_path: Path,
    object_id: int | None = None,
    object_name: str | None = "Workpiece - Ring",
) -> Mesh:
    with key_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    names = scan_object_names(lines)
    target_id = choose_object_id(names, object_id, object_name)
    target_name = names.get(target_id, "")

    node_ids: np.ndarray | None = None
    coordinates: np.ndarray | None = None
    elements: np.ndarray | None = None

    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if not parts:
            i += 1
            continue

        keyword = parts[0].upper()

        if keyword == "RZ" and len(parts) >= 3:
            oid = int(parts[1])
            count = int(parts[2])
            if oid == target_id:
                if node_ids is not None:
                    raise ValueError(f"对象 {target_id} 出现了多个 RZ 数据块。")
                node_ids, coordinates, i = read_rz_block(lines, i + 1, count)
                continue

        if keyword == "ELMCON" and len(parts) >= 4:
            oid = int(parts[1])
            element_count = int(parts[2])
            nodes_per_element = int(parts[3])
            if oid == target_id:
                if elements is not None:
                    raise ValueError(f"对象 {target_id} 出现了多个 ELMCON 数据块。")
                elements, i = read_elmcon_block(
                    lines, i + 1, element_count, nodes_per_element
                )
                continue

        i += 1

    if node_ids is None or coordinates is None:
        raise ValueError(f"对象 {target_id} 未找到 RZ 节点坐标数据块。")
    if elements is None:
        raise ValueError(f"对象 {target_id} 未找到 ELMCON 单元连接数据块。")

    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("RZ 数据中存在重复节点编号。")

    mesh = Mesh(
        object_id=target_id,
        object_name=target_name,
        node_ids=node_ids,
        coordinates=coordinates,
        elements=elements,
    )
    validate_mesh(mesh)
    return mesh


def validate_mesh(mesh: Mesh) -> None:
    known = set(int(v) for v in mesh.node_ids)
    referenced = set(int(v) for v in np.unique(mesh.elements))
    missing = referenced - known
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"ELMCON 引用了 RZ 中不存在的节点，共 {len(missing)} 个，示例：{preview}"
        )


def build_node_index(mesh: Mesh) -> Dict[int, int]:
    return {int(node_id): i for i, node_id in enumerate(mesh.node_ids)}


def extract_boundary_faces(elements: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    六面体每个面会被相邻单元共享：
    - 出现两次：内部面
    - 只出现一次：外边界面
    """
    face_map: Dict[Tuple[int, int, int, int], Tuple[int, Tuple[int, int, int, int]]] = {}

    for element in elements:
        for local_face in HEX8_FACES:
            face = tuple(int(element[i]) for i in local_face)
            key = tuple(sorted(face))
            if key in face_map:
                count, stored_face = face_map[key] # type: ignore
                face_map[key] = (count + 1, stored_face) # type: ignore
            else:
                face_map[key] = (1, face) # type: ignore

    non_manifold = [key for key, (count, _) in face_map.items() if count > 2]
    if non_manifold:
        print(
            f"警告：检测到 {len(non_manifold)} 个被三个及以上单元共享的非流形面。",
            file=sys.stderr,
        )

    return [face for count, face in face_map.values() if count == 1]


def deduplicate_points(points: Iterable[np.ndarray], tol: float) -> List[np.ndarray]:
    result: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for p in points:
        key = tuple(np.rint(np.asarray(p) / tol).astype(np.int64))
        result[key].append(np.asarray(p, dtype=float))
    return [np.mean(values, axis=0) for values in result.values()]


def farthest_pair(points: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d2 = float(np.sum((points[i] - points[j]) ** 2))
            if d2 > best_d2:
                best_d2 = d2
                best_i, best_j = i, j
    return points[best_i], points[best_j]


def intersect_face_with_plane(
    face_xyz: np.ndarray,
    plane_z: float,
    z_tol: float,
    merge_tol: float,
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    返回：
      ("regular", [segment])：普通相交
      ("coplanar", [四条边])：整个四边形位于截面平面
      ("none", [])：不相交
    """
    distances = face_xyz[:, 2] - plane_z
    on_plane = np.abs(distances) <= z_tol

    if np.all(on_plane):
        edges = []
        for i in range(4):
            a = face_xyz[i, :2].copy()
            b = face_xyz[(i + 1) % 4, :2].copy()
            if np.linalg.norm(a - b) > merge_tol:
                edges.append((a, b))
        return "coplanar", edges

    intersections: List[np.ndarray] = []
    for i in range(4):
        p = face_xyz[i]
        q = face_xyz[(i + 1) % 4]
        dp = p[2] - plane_z
        dq = q[2] - plane_z
        p_on = abs(dp) <= z_tol
        q_on = abs(dq) <= z_tol

        if p_on:
            intersections.append(p[:2].copy())

        if (dp < -z_tol and dq > z_tol) or (dp > z_tol and dq < -z_tol):
            t = dp / (dp - dq)
            point = p + t * (q - p)
            intersections.append(point[:2].copy())

        # 若整条边位于平面上，其两个端点都会在相邻循环中加入，
        # 后续统一去重。

    unique_points = deduplicate_points(intersections, merge_tol)

    if len(unique_points) < 2:
        return "none", []

    # 对平面四边形，交集应为一条线段；当截面恰好经过顶点或面有轻微翘曲时，
    # 可能得到 3~4 个候选点，此时取最远两点作为线段端点。
    a, b = farthest_pair(unique_points)
    if np.linalg.norm(a - b) <= merge_tol:
        return "none", []
    return "regular", [(a, b)]


def canonical_segment_key(
    a: np.ndarray,
    b: np.ndarray,
    tol: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    ka = tuple(np.rint(a / tol).astype(np.int64))
    kb = tuple(np.rint(b / tol).astype(np.int64))
    return (ka, kb) if ka <= kb else (kb, ka)


def extract_section_segments(
    mesh: Mesh,
    boundary_faces: Sequence[Tuple[int, int, int, int]],
    plane_z: float,
    z_tol: float,
    merge_tol: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    node_index = build_node_index(mesh)

    regular_segments: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]], Tuple[np.ndarray, np.ndarray]
    ] = {}

    # 若截面与外表面的一片四边形完全重合，需要对这些共面面的边进行计数；
    # 共面区域内部边出现两次，区域边界边只出现一次。
    coplanar_edge_counts: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[int, np.ndarray, np.ndarray],
    ] = {}

    for face in boundary_faces:
        try:
            face_xyz = np.asarray(
                [mesh.coordinates[node_index[node_id]] for node_id in face],
                dtype=float,
            )
        except KeyError as exc:
            raise ValueError(f"边界面引用了不存在的节点：{exc.args[0]}") from exc

        kind, segments = intersect_face_with_plane(
            face_xyz, plane_z, z_tol, merge_tol
        )

        if kind == "regular":
            for a, b in segments:
                key = canonical_segment_key(a, b, merge_tol)
                regular_segments.setdefault(key, (a, b))

        elif kind == "coplanar":
            for a, b in segments:
                key = canonical_segment_key(a, b, merge_tol)
                if key in coplanar_edge_counts:
                    count, old_a, old_b = coplanar_edge_counts[key]
                    coplanar_edge_counts[key] = (count + 1, old_a, old_b)
                else:
                    coplanar_edge_counts[key] = (1, a, b)

    # 共面区域只保留其外周边；再与普通截线合并并去重。
    for key, (count, a, b) in coplanar_edge_counts.items():
        if count == 1:
            regular_segments.setdefault(key, (a, b))

    return list(regular_segments.values())


def build_section_components(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    merge_tol: float,
) -> Tuple[
    Dict[Tuple[int, int], np.ndarray],
    List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
]:
    if not segments:
        raise ValueError("截面未提取到任何线段，请检查 plane-z、单位和容差。")

    point_accumulator: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    edges_set = set()

    for a, b in segments:
        ka = tuple(np.rint(a / merge_tol).astype(np.int64))
        kb = tuple(np.rint(b / merge_tol).astype(np.int64))
        if ka == kb:
            continue
        point_accumulator[ka].append(np.asarray(a, dtype=float))
        point_accumulator[kb].append(np.asarray(b, dtype=float))
        edge = (ka, kb) if ka <= kb else (kb, ka)
        edges_set.add(edge)

    points = {key: np.mean(values, axis=0) for key, values in point_accumulator.items()}

    adjacency: Dict[Tuple[int, int], set] = defaultdict(set)
    for a, b in edges_set:
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited = set()
    components: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = []

    for start in adjacency:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component_nodes = set()

        while queue:
            node = queue.popleft()
            component_nodes.add(node)
            for nb in adjacency[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        component_edges = [
            edge for edge in edges_set if edge[0] in component_nodes
        ]
        components.append(component_edges)

    return points, components


def component_perimeter(
    component_edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    point_map: Dict[Tuple[int, int], np.ndarray],
) -> float:
    return float(
        sum(np.linalg.norm(point_map[a] - point_map[b]) for a, b in component_edges)
    )


def algebraic_circle_center(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    b = x * x + y * y
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    return solution[:2]


def order_component_as_loop(
    component_edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    point_map: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[np.ndarray, bool]:
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for a, b in component_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    nodes = list(adjacency)
    degrees = [len(adjacency[node]) for node in nodes]
    is_degree_two_loop = all(degree == 2 for degree in degrees)

    if is_degree_two_loop:
        start = min(nodes)
        path = [start]
        previous = None
        current = start

        while True:
            neighbors = adjacency[current]
            next_node = neighbors[0] if neighbors[0] != previous else neighbors[1]

            if next_node == start:
                break
            if next_node in path:
                is_degree_two_loop = False
                break

            path.append(next_node)
            previous, current = current, next_node

        if is_degree_two_loop and len(path) == len(nodes):
            return np.asarray([point_map[node] for node in path]), True

    # 退化情形：截面恰好经过多个网格顶点时，图节点度数可能不为 2。
    # 对近圆形轮廓，以拟合圆心为极点按角度排序作为回退方案。
    raw_points = np.asarray([point_map[node] for node in nodes])
    center = algebraic_circle_center(raw_points)
    angles = np.arctan2(raw_points[:, 1] - center[1], raw_points[:, 0] - center[0])
    order = np.argsort(angles)
    return raw_points[order], False


def polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def closed_polyline_perimeter(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))


def resample_closed_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    if sample_count < 16:
        raise ValueError("每条轮廓的采样点数至少应为 16。")

    next_points = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(next_points - points, axis=1)
    valid = lengths > np.finfo(float).eps
    if np.count_nonzero(valid) < 3:
        raise ValueError("截面轮廓退化，无法进行闭合曲线重采样。")

    starts = points[valid]
    ends = next_points[valid]
    lengths = lengths[valid]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]

    targets = np.linspace(0.0, total, sample_count, endpoint=False)
    segment_indices = np.searchsorted(cumulative, targets, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)

    local = (targets - cumulative[segment_indices]) / lengths[segment_indices]
    samples = starts[segment_indices] + local[:, None] * (
        ends[segment_indices] - starts[segment_indices]
    )
    return samples


def fit_least_squares_circle(points: np.ndarray) -> CircleResult:
    center0 = algebraic_circle_center(points)

    try:
        from scipy.optimize import least_squares

        def residual(center: np.ndarray) -> np.ndarray:
            radii = np.linalg.norm(points - center, axis=1)
            return radii - np.mean(radii)

        optimized = least_squares(
            residual,
            center0,
            method="lm" if len(points) >= 3 else "trf",
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=3000,
        )
        center = optimized.x
    except ImportError:
        center = center0

    radii = np.linalg.norm(points - center, axis=1)
    r_min = float(np.min(radii))
    r_max = float(np.max(radii))
    radius = float(np.mean(radii))

    return CircleResult(
        method="LSC",
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=radius,
        roundness=r_max - r_min,
        r_min=r_min,
        r_max=r_max,
    )


def fit_minimum_zone_circle(
    points: np.ndarray,
    lsc: CircleResult,
) -> CircleResult | None:
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError:
        return None

    initial = np.array([lsc.center_x, lsc.center_y], dtype=float)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    radius_scale = max(abs(lsc.radius), span * 0.5, 1.0)
    search_half_width = max(
        0.08 * radius_scale,
        10.0 * lsc.roundness,
        1e-6,
    )

    def objective(center: np.ndarray) -> float:
        radii = np.linalg.norm(points - center, axis=1)
        return float(np.max(radii) - np.min(radii))

    bounds = [
        (initial[0] - search_half_width, initial[0] + search_half_width),
        (initial[1] - search_half_width, initial[1] + search_half_width),
    ]

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=0, # type: ignore
        popsize=12,
        maxiter=250,
        tol=1e-10,
        atol=1e-12,
        polish=False,
        updating="immediate",
        workers=1,
    )

    local_result = minimize(
        objective,
        global_result.x,
        method="Nelder-Mead",
        options={
            "xatol": 1e-12,
            "fatol": 1e-12,
            "maxiter": 5000,
        },
    )

    center = local_result.x if local_result.fun <= global_result.fun else global_result.x
    radii = np.linalg.norm(points - center, axis=1)
    r_min = float(np.min(radii))
    r_max = float(np.max(radii))

    return CircleResult(
        method="MZC",
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=0.5 * (r_min + r_max),
        roundness=r_max - r_min,
        r_min=r_min,
        r_max=r_max,
    )


def analyze_profile(name: str, loop: np.ndarray, samples: int) -> ProfileResult:
    resampled = resample_closed_polyline(loop, samples)
    lsc = fit_least_squares_circle(resampled)
    mzc = fit_minimum_zone_circle(resampled, lsc)

    return ProfileResult(
        name=name,
        raw_vertex_count=len(loop),
        sample_count=len(resampled),
        perimeter=closed_polyline_perimeter(loop),
        polygon_area=abs(polygon_signed_area(loop)),
        least_squares=lsc,
        minimum_zone=mzc,
    )


def circle_xy(center_x: float, center_y: float, radius: float, n: int = 720):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    return center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)


def save_plot(
    output_path: Path,
    plane_z: float,
    profiles: Sequence[Tuple[str, np.ndarray, ProfileResult]],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))

    for name, loop, result in profiles:
        closed = np.vstack([loop, loop[0]])
        line = ax.plot(closed[:, 0], closed[:, 1], linewidth=1.1, label=f"{name} section profile")[0]

        circle = result.minimum_zone or result.least_squares
        x_mid, y_mid = circle_xy(circle.center_x, circle.center_y, circle.radius)
        ax.plot(
            x_mid,
            y_mid,
            linestyle="--",
            linewidth=1.0,
            label=f"{name}-{circle.method} mean circle",
        )

        if result.minimum_zone is not None:
            x_min, y_min = circle_xy(circle.center_x, circle.center_y, circle.r_min)
            x_max, y_max = circle_xy(circle.center_x, circle.center_y, circle.r_max)
            ax.plot(x_min, y_min, linestyle=":", linewidth=0.8)
            ax.plot(x_max, y_max, linestyle=":", linewidth=0.8)

        ax.scatter(
            [circle.center_x],
            [circle.center_y],
            marker="+",
            s=45,
            label=f"{name} center",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Workpiece-Ring section at z={plane_z:g}")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_results_csv(
    output_path: Path,
    mesh: Mesh,
    plane_z: float,
    boundary_face_count: int,
    segment_count: int,
    results: Sequence[ProfileResult],
) -> None:
    fields = [
        "object_id",
        "object_name",
        "node_count",
        "element_count",
        "boundary_face_count",
        "section_segment_count",
        "plane_z",
        "profile",
        "raw_vertex_count",
        "resample_count",
        "perimeter",
        "polygon_area",
        "method",
        "center_x",
        "center_y",
        "radius",
        "r_min",
        "r_max",
        "roundness",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for profile in results:
            circle_results = [profile.least_squares]
            if profile.minimum_zone is not None:
                circle_results.append(profile.minimum_zone)

            for circle in circle_results:
                writer.writerow(
                    {
                        "object_id": mesh.object_id,
                        "object_name": mesh.object_name,
                        "node_count": len(mesh.node_ids),
                        "element_count": len(mesh.elements),
                        "boundary_face_count": boundary_face_count,
                        "section_segment_count": segment_count,
                        "plane_z": plane_z,
                        "profile": profile.name,
                        "raw_vertex_count": profile.raw_vertex_count,
                        "resample_count": profile.sample_count,
                        "perimeter": profile.perimeter,
                        "polygon_area": profile.polygon_area,
                        "method": circle.method,
                        "center_x": circle.center_x,
                        "center_y": circle.center_y,
                        "radius": circle.radius,
                        "r_min": circle.r_min,
                        "r_max": circle.r_max,
                        "roundness": circle.roundness,
                    }
                )


def print_summary(
    mesh: Mesh,
    boundary_face_count: int,
    segment_count: int,
    component_count: int,
    results: Sequence[ProfileResult],
) -> None:
    print("=" * 72)
    print(f"对象：{mesh.object_name!r}（ID={mesh.object_id}）")
    print(f"节点数量：{len(mesh.node_ids)}")
    print(f"单元数量：{len(mesh.elements)}")
    print(f"外边界面数量：{boundary_face_count}")
    print(f"截面线段数量：{segment_count}")
    print(f"截面连通分量数量：{component_count}")
    print("=" * 72)

    for result in results:
        print(f"[{result.name}]")
        print(f"  原始轮廓顶点数：{result.raw_vertex_count}")
        print(f"  轮廓周长：{result.perimeter:.10g}")
        print(f"  包围面积：{result.polygon_area:.10g}")

        lsc = result.least_squares
        print(
            "  最小二乘圆 LSC："
            f"圆心=({lsc.center_x:.10g}, {lsc.center_y:.10g}), "
            f"半径={lsc.radius:.10g}, "
            f"径向峰谷值={lsc.roundness:.10g}"
        )

        if result.minimum_zone is not None:
            mzc = result.minimum_zone
            print(
                "  最小区域圆 MZC："
                f"圆心=({mzc.center_x:.10g}, {mzc.center_y:.10g}), "
                f"Rmin={mzc.r_min:.10g}, "
                f"Rmax={mzc.r_max:.10g}, "
                f"圆度={mzc.roundness:.10g}"
            )
        else:
            print("  未安装 scipy，未计算最小区域圆 MZC。")

    print("=" * 72)


def determine_default_tolerances(mesh: Mesh) -> Tuple[float, float]:
    ranges = np.ptp(mesh.coordinates, axis=0)
    characteristic_length = max(float(np.max(ranges)), 1.0)
    # z 方向判定容差与二维端点合并容差分别给出。
    return characteristic_length * 1e-9, characteristic_length * 1e-8


def main() -> int:
    parser = argparse.ArgumentParser(
        description="读取 DEFORM KEY 网格，提取 Ring 的平面截面并计算内外轮廓圆度。"
    )
    parser.add_argument("key_file", type=Path,
                        nargs="?",
                        default=Path(r"/Users/hmr/Desktop/Multi-obj-optimism/data/KEY_FILE/RINGROLL.KEY"), 
                        help="DEFORM .KEY 文件路径")
    parser.add_argument(
        "--object-name",
        default="Workpiece - Ring",
        help='对象名称，默认："Workpiece - Ring"',
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="直接指定对象编号；给出后优先于 --object-name",
    )
    parser.add_argument(
        "--plane-z",
        type=float,
        default=0.0,
        help="截面平面的 z 值，默认 0",
    )
    parser.add_argument(
        "--z-tol",
        type=float,
        default=None,
        help="判断节点是否位于截面平面的绝对容差；默认按模型尺度自动确定",
    )
    parser.add_argument(
        "--merge-tol",
        type=float,
        default=None,
        help="合并截面端点的 XY 绝对容差；默认按模型尺度自动确定",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="每条闭合轮廓按弧长均匀重采样的点数，默认 3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认与 KEY 文件同目录",
    )
    args = parser.parse_args()

    if not args.key_file.exists():
        parser.error(f"文件不存在：{args.key_file}")

    output_dir = args.output_dir or args.key_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = read_deform_key(
        args.key_file,
        object_id=args.object_id,
        object_name=args.object_name,
    )

    default_z_tol, default_merge_tol = determine_default_tolerances(mesh)
    z_tol = args.z_tol if args.z_tol is not None else default_z_tol
    merge_tol = args.merge_tol if args.merge_tol is not None else default_merge_tol

    if z_tol <= 0 or merge_tol <= 0:
        raise ValueError("z-tol 和 merge-tol 必须为正数。")

    boundary_faces = extract_boundary_faces(mesh.elements)
    segments = extract_section_segments(
        mesh,
        boundary_faces,
        plane_z=args.plane_z,
        z_tol=z_tol,
        merge_tol=merge_tol,
    )

    point_map, components = build_section_components(segments, merge_tol)

    component_infos = []
    for component in components:
        if len(component) < 3:
            continue
        perimeter = component_perimeter(component, point_map)
        loop, exact_graph_loop = order_component_as_loop(component, point_map)
        if len(loop) < 3:
            continue
        area = abs(polygon_signed_area(loop))
        component_infos.append(
            {
                "component": component,
                "perimeter": perimeter,
                "loop": loop,
                "area": area,
                "exact_graph_loop": exact_graph_loop,
            }
        )

    if len(component_infos) < 2:
        raise ValueError(
            "未获得两个有效的闭合截面轮廓。"
            f"当前有效连通分量数：{len(component_infos)}。"
            "请检查 z=0 是否确实与 Ring 相交，并适当调整 --z-tol 和 --merge-tol。"
        )

    # 截面可能包含极小伪分量，优先取周长最大的两个主轮廓。
    component_infos.sort(key=lambda item: item["perimeter"], reverse=True)
    selected = component_infos[:2]

    if len(component_infos) > 2:
        ignored = len(component_infos) - 2
        print(
            f"警告：截面检测到 {len(component_infos)} 个有效分量，"
            f"仅使用周长最大的两个，忽略 {ignored} 个较小分量。",
            file=sys.stderr,
        )

    for item in selected:
        if not item["exact_graph_loop"]:
            print(
                "警告：某截面分量的节点度数不全为 2，已采用绕拟合圆心的角度排序。",
                file=sys.stderr,
            )

    # 按包围面积区分内圈和外圈。
    selected.sort(key=lambda item: item["area"])
    inner_loop = selected[0]["loop"]
    outer_loop = selected[1]["loop"]

    inner_result = analyze_profile("内圈", inner_loop, args.samples)
    outer_result = analyze_profile("外圈", outer_loop, args.samples)
    results = [inner_result, outer_result]

    stem = args.key_file.stem
    csv_path = output_dir / f"{stem}_z{args.plane_z:g}_roundness.csv"
    png_path = output_dir / f"{stem}_z{args.plane_z:g}_section.png"

    write_results_csv(
        csv_path,
        mesh,
        args.plane_z,
        len(boundary_faces),
        len(segments),
        results,
    )
    save_plot(
        png_path,
        args.plane_z,
        [
            ("Inner", inner_loop, inner_result),
            ("Outer", outer_loop, outer_result),
        ],
    )

    print_summary(
        mesh,
        len(boundary_faces),
        len(segments),
        len(components),
        results,
    )
    print(f"结果表：{csv_path}")
    print(f"截面图：{png_path}")
    print(f"使用容差：z_tol={z_tol:.6g}, merge_tol={merge_tol:.6g}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
