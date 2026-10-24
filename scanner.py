"""文档扫描预处理核心算法模块。

职责：图像解码、文档四边形轮廓检测、角点排序、透视矫正、图像增强。
本模块不依赖 Streamlit，可独立测试与复用。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 输入图片最长边限制，防止超大图片占用过多内存
MAX_IMAGE_DIM = 2000
# 矫正输出图片最长边限制
MAX_OUTPUT_DIM = 2500
# 输出最小边长
MIN_OUTPUT_DIM = 16
# 上传文件大小上限（字节）
MAX_FILE_BYTES = 20 * 1024 * 1024


class ScanError(Exception):
    """扫描处理中的可预期错误，界面应捕获并以中文提示展示。"""


@dataclass
class DetectionResult:
    """文档四角自动检测结果。"""

    found: bool
    corners: np.ndarray | None = None  # (4, 2) float32，顺序：左上、右上、右下、左下
    edge_image: np.ndarray | None = None
    message: str = ""
    candidates: int = 0


# ---------------------------------------------------------------------------
# 图像读取与尺寸控制
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """从字节流解码图像（对中文文件名安全，因为不经过文件路径）。

    返回 BGR 三通道 uint8 图像；灰度图会被解码为三通道。
    """
    if not data:
        raise ScanError("文件为空，请重新选择图片。")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ScanError("无法解码图片，文件可能已损坏或格式不受支持。")
    return img


def resize_to_max(img: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> tuple[np.ndarray, float]:
    """等比缩小到最长边不超过 max_dim。返回 (图像, 缩放比例)。"""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img, 1.0
    scale = max_dim / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


# ---------------------------------------------------------------------------
# 角点排序与校验
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """将四个点排序为：左上、右上、右下、左下。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.array(
        [
            pts[np.argmin(s)],  # 左上：x+y 最小
            pts[np.argmin(d)],  # 右上：y-x 最小
            pts[np.argmax(s)],  # 右下：x+y 最大
            pts[np.argmax(d)],  # 左下：y-x 最大
        ],
        dtype=np.float32,
    )
    return ordered


def validate_corners(pts: np.ndarray, min_dist: float = 5.0) -> np.ndarray:
    """校验四点：必须是 4 个点、无重复点、能构成非退化四边形。"""
    rect = order_points(pts)
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(rect[i] - rect[j]) < min_dist:
                raise ScanError("四个角点中存在距离过近或重复的点，请重新选择。")
    area = abs(cv2.contourArea(rect.reshape(4, 1, 2)))
    if area < 100:
        raise ScanError("四个角点构成的四边形面积过小，无法矫正。")
    return rect


# ---------------------------------------------------------------------------
# 文档四边形自动检测
# ---------------------------------------------------------------------------

def _is_full_frame(approx: np.ndarray, shape: tuple[int, int], margin_ratio: float = 0.02) -> bool:
    """判断四边形是否几乎贴满整张照片（即误检为照片边框）。"""
    h, w = shape
    pts = approx.reshape(4, 2)
    mx, my = w * margin_ratio, h * margin_ratio
    near_border = (
        ((pts[:, 0] < mx) | (pts[:, 0] > w - mx)).all()
        and ((pts[:, 1] < my) | (pts[:, 1] > h - my)).all()
    )
    return bool(near_border)


def _find_quad_in_edges(
    edges: np.ndarray,
    img_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float = 0.98,
) -> tuple[np.ndarray | None, int]:
    """在边缘图中筛选符合条件的四边形轮廓。"""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    img_area = float(img_shape[0] * img_shape[1])
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    checked = 0
    for cnt in contours[:8]:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio:
            break  # 后续轮廓面积更小，不必再看
        if area > img_area * max_area_ratio:
            continue
        checked += 1
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if _is_full_frame(approx, img_shape):
            continue
        return order_points(approx.reshape(4, 2)), checked
    return None, checked


def detect_document(
    img: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    min_area_ratio: float = 0.2,
    blur_ksize: int = 5,
) -> DetectionResult:
    """自动检测文档四角。

    流程：灰度化 -> 高斯降噪 -> Canny 边缘 -> 形态学闭运算连接断边 ->
    轮廓检测 -> 四边形/凸性/面积占比筛选。使用多组 Canny 阈值重试以提高成功率。
    """
    if img is None or img.size == 0:
        return DetectionResult(False, message="输入图像为空。")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 多组阈值尝试，兼顾对比度强弱的场景
    threshold_sets = [(canny_low, canny_high), (30, 100), (75, 200)]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    best_edges = None
    total_checked = 0
    for low, high in threshold_sets:
        low = int(max(1, min(low, high - 1)))
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)
        if best_edges is None:
            best_edges = edges
        corners, checked = _find_quad_in_edges(edges, (h, w), min_area_ratio)
        total_checked += checked
        if corners is not None:
            return DetectionResult(
                True,
                corners=corners,
                edge_image=edges,
                message=f"检测成功（Canny 阈值 {low}/{high}）。",
                candidates=total_checked,
            )
    return DetectionResult(
        False,
        edge_image=best_edges,
        message="自动检测失败：未找到符合条件的四边形文档轮廓，请使用手动四点调整。",
        candidates=total_checked,
    )


# ---------------------------------------------------------------------------
# 透视矫正
# ---------------------------------------------------------------------------

def four_point_transform(
    img: np.ndarray,
    pts: np.ndarray,
    max_dim: int = MAX_OUTPUT_DIM,
) -> np.ndarray:
    """根据四个角点执行透视矫正，输出正视文档图像。"""
    if img is None or img.size == 0:
        raise ScanError("输入图像为空，无法执行透视矫正。")
    rect = validate_corners(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(round(max(width_a, width_b)))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(round(max(height_a, height_b)))

    # 尺寸校验与限制，避免异常内存占用
    max_width = int(np.clip(max_width, MIN_OUTPUT_DIM, max_dim))
    max_height = int(np.clip(max_height, MIN_OUTPUT_DIM, max_dim))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    if not np.isfinite(M).all():
        raise ScanError("透视变换矩阵计算失败，请检查四个角点是否合理。")
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    if warped.size == 0:
        raise ScanError("透视矫正输出为空，请检查四个角点。")
    return warped


# ---------------------------------------------------------------------------
# 图像增强（去噪 / 去阴影 / 二值化）
# ---------------------------------------------------------------------------

def remove_shadow(gray: np.ndarray, bg_ksize: int = 51) -> np.ndarray:
    """基于背景估计的去阴影：用大核中值滤波估计光照背景，再做除法归一。"""
    bg_ksize = int(bg_ksize)
    if bg_ksize % 2 == 0:
        bg_ksize += 1
    bg_ksize = max(3, bg_ksize)
    background = cv2.medianBlur(gray, bg_ksize)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized


def enhance(
    img: np.ndarray,
    mode: str = "gray",
    denoise: bool = True,
    bg_ksize: int = 51,
    block_size: int = 41,
    threshold_c: int = 10,
) -> np.ndarray:
    """图像增强。

    mode="gray"：去噪 + 去阴影后的灰度扫描图。
    mode="bw"：在灰度增强基础上做自适应二值化，输出黑白扫描图。
    返回单通道 uint8 图像。
    """
    if img is None or img.size == 0:
        raise ScanError("输入图像为空，无法增强。")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    flattened = remove_shadow(gray, bg_ksize)

    if mode == "gray":
        return flattened
    if mode == "bw":
        block_size = int(block_size)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)
        # 块大小不能超过图像尺寸
        block_size = min(block_size, min(flattened.shape[:2]) | 1)
        if block_size < 3:
            block_size = 3
        binary = cv2.adaptiveThreshold(
            flattened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(threshold_c),
        )
        return binary
    raise ScanError(f"未知的增强模式：{mode}")


# ---------------------------------------------------------------------------
# 辅助：编码与可视化
# ---------------------------------------------------------------------------

def encode_png(img: np.ndarray) -> bytes:
    """将图像编码为 PNG 字节流，用于下载。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ScanError("PNG 编码失败。")
    return buf.tobytes()


def draw_corners(img: np.ndarray, corners: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """在图像上绘制四边形与角点，用于预览（返回 RGB 图）。"""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    rect = order_points(corners).astype(int)
    cv2.polylines(vis, [rect.reshape(4, 1, 2)], True, color, 3)
    labels = ["左上", "右上", "右下", "左下"]
    for i, (x, y) in enumerate(rect):
        cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.putText(
            vis, str(i + 1), (int(x) + 12, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def draw_points(img: np.ndarray, points) -> np.ndarray:
    """绘制 1~4 个已选角点（不足 4 个点时也可见），用于手动取点预览。返回 RGB。"""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 4:
        rect = order_points(pts).astype(int)
        cv2.polylines(vis, [rect.reshape(4, 1, 2)], True, (0, 0, 255), 3)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.putText(
            vis, str(i + 1), (int(x) + 12, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR/灰度 -> RGB，用于 Streamlit 显示。"""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
