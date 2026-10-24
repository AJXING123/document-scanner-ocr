"""scanner 核心算法测试。

使用程序化合成的文档图片，不依赖外部网络资源。
"""
import cv2
import numpy as np
import pytest

import scanner


# ---------------------------------------------------------------------------
# 测试图片合成工具
# ---------------------------------------------------------------------------

def make_tilted_document(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """合成一张放在杂乱桌面上的倾斜文档照片。返回 (图像, 文档四角真值)。"""
    rng = np.random.default_rng(seed)
    h, w = 800, 1000
    # 桌面背景：低对比度平滑噪声，避免产生大轮廓
    bg = rng.integers(80, 140, (h, w, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 15)

    # 文档：白纸 + 模拟文字行
    doc_h, doc_w = 640, 480
    doc = np.full((doc_h, doc_w, 3), 245, np.uint8)
    for y in range(60, doc_h - 40, 32):
        x_end = doc_w - 40 - int(rng.integers(0, 100))
        cv2.line(doc, (40, y), (x_end, y), (30, 30, 30), 3)

    # 透视投影到背景上（倾斜放置）
    src = np.array([[0, 0], [doc_w - 1, 0], [doc_w - 1, doc_h - 1], [0, doc_h - 1]], np.float32)
    dst = np.array([[220, 90], [760, 140], [700, 690], [150, 600]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped_doc = cv2.warpPerspective(doc, M, (w, h))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(int), 255)
    out = np.where(cv2.merge([mask] * 3) > 0, warped_doc, bg)
    return out, dst


def make_non_document() -> np.ndarray:
    """合成一张没有文档的普通场景图（平滑渐变 + 小圆点）。"""
    h, w = 600, 800
    grad = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)
    img = cv2.merge([grad] * 3)
    rng = np.random.default_rng(1)
    for _ in range(30):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, center, int(rng.integers(5, 25)), (200,), -1)
    return cv2.GaussianBlur(img, (9, 9), 0)


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 图像读取
# ---------------------------------------------------------------------------

def test_load_color_image():
    img, _ = make_tilted_document()
    loaded = scanner.load_image_from_bytes(to_png_bytes(img))
    assert loaded.ndim == 3 and loaded.shape[2] == 3
    assert loaded.dtype == np.uint8


def test_load_grayscale_image():
    gray = np.full((300, 400), 180, np.uint8)
    loaded = scanner.load_image_from_bytes(to_png_bytes(gray))
    # 统一解码为三通道 BGR，保证后续流程一致
    assert loaded.ndim == 3 and loaded.shape[2] == 3


def test_load_empty_bytes_raises():
    with pytest.raises(scanner.ScanError):
        scanner.load_image_from_bytes(b"")


def test_load_invalid_bytes_raises():
    with pytest.raises(scanner.ScanError):
        scanner.load_image_from_bytes(b"this is not an image")


def test_resize_to_max():
    big = np.zeros((4000, 5000, 3), np.uint8)
    resized, scale = scanner.resize_to_max(big, 2000)
    assert max(resized.shape[:2]) == 2000
    assert scale == pytest.approx(0.4)
    small = np.zeros((300, 400, 3), np.uint8)
    same, scale2 = scanner.resize_to_max(small, 2000)
    assert scale2 == 1.0 and same.shape == small.shape


# ---------------------------------------------------------------------------
# 角点排序与校验
# ---------------------------------------------------------------------------

def test_order_points():
    quad = np.array([[100, 100], [300, 120], [280, 400], [90, 380]], np.float32)
    shuffled = quad[[2, 0, 3, 1]]
    ordered = scanner.order_points(shuffled)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0] and tl[1] < bl[1]
    assert br[0] > bl[0] and br[1] > tr[1]
    # 与真值逐一对应（允许浮点误差）
    for p in quad:
        assert np.min(np.linalg.norm(ordered - p, axis=1)) < 1e-3


def test_validate_duplicate_points_raises():
    pts = np.array([[10, 10], [10, 10], [300, 300], [50, 300]], np.float32)
    with pytest.raises(scanner.ScanError):
        scanner.validate_corners(pts)


# ---------------------------------------------------------------------------
# 文档自动检测
# ---------------------------------------------------------------------------

def test_detect_tilted_document_success():
    img, dst = make_tilted_document()
    result = scanner.detect_document(img)
    assert result.found, result.message
    assert result.corners.shape == (4, 2)
    # 检测角点应与真值接近（允许边缘检测带来的像素误差）
    for p in dst:
        assert np.min(np.linalg.norm(result.corners - p, axis=1)) < 40


def test_detect_grayscale_input():
    img, _ = make_tilted_document()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = scanner.detect_document(gray)
    assert result.found, result.message


def test_detect_non_document_fails_gracefully():
    img = make_non_document()
    result = scanner.detect_document(img)
    # 允许失败进入手动流程，不允许崩溃
    assert isinstance(result, scanner.DetectionResult)
    assert not result.found
    assert result.message  # 必须有明确中文提示


def test_detect_empty_input():
    result = scanner.detect_document(np.zeros((0, 0, 3), np.uint8))
    assert not result.found


# ---------------------------------------------------------------------------
# 透视矫正
# ---------------------------------------------------------------------------

def test_warp_output_size_valid():
    img, dst = make_tilted_document()
    warped = scanner.four_point_transform(img, dst)
    h, w = warped.shape[:2]
    assert warped.dtype == np.uint8
    assert scanner.MIN_OUTPUT_DIM <= w <= scanner.MAX_OUTPUT_DIM
    assert scanner.MIN_OUTPUT_DIM <= h <= scanner.MAX_OUTPUT_DIM
    # 输出尺寸按四条投影边长估算，应接近投影四边形的实际跨度且不退化
    assert max(w, h) >= 400
    assert 0.5 <= w / h <= 2.0


def test_warp_with_detected_corners():
    img, _ = make_tilted_document()
    result = scanner.detect_document(img)
    assert result.found
    warped = scanner.four_point_transform(img, result.corners)
    assert warped.size > 0


def test_warp_invalid_points_raises():
    img, _ = make_tilted_document()
    with pytest.raises(scanner.ScanError):
        scanner.four_point_transform(img, np.array([[5, 5], [6, 5], [5, 6], [6, 6]], np.float32))


# ---------------------------------------------------------------------------
# 图像增强
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["gray", "bw"])
def test_enhance_output_shape_and_dtype(mode):
    img, dst = make_tilted_document()
    warped = scanner.four_point_transform(img, dst)
    enhanced = scanner.enhance(warped, mode=mode)
    assert enhanced.dtype == np.uint8
    assert enhanced.ndim == 2
    assert enhanced.shape == warped.shape[:2]


def test_enhance_bw_is_binary():
    img, dst = make_tilted_document()
    warped = scanner.four_point_transform(img, dst)
    enhanced = scanner.enhance(warped, mode="bw")
    unique = set(np.unique(enhanced).tolist())
    assert unique.issubset({0, 255})


def test_enhance_bad_mode_raises():
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(scanner.ScanError):
        scanner.enhance(img, mode="unknown")


def test_enhance_small_image_bw():
    # 小图上二值化块大小必须被安全收缩，不能崩溃
    img = np.full((30, 30, 3), 200, np.uint8)
    out = scanner.enhance(img, mode="bw", block_size=41)
    assert out.shape == (30, 30)


# ---------------------------------------------------------------------------
# 编码输出
# ---------------------------------------------------------------------------

def test_png_encode_roundtrip():
    img, dst = make_tilted_document()
    warped = scanner.four_point_transform(img, dst)
    enhanced = scanner.enhance(warped, mode="bw")
    data = scanner.encode_png(enhanced)
    assert isinstance(data, bytes) and len(data) > 0
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    assert decoded.shape == enhanced.shape


def test_draw_corners_returns_rgb():
    img, dst = make_tilted_document()
    vis = scanner.draw_corners(img, dst)
    assert vis.ndim == 3 and vis.shape[2] == 3
    assert vis.shape[:2] == img.shape[:2]
