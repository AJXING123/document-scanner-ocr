"""文档扫描预处理工具（OCR 前置）— Streamlit 界面。

功能：上传图片 -> 自动检测文档四角 -> （可选）手动四点调整 ->
透视矫正 -> 灰度/黑白增强 -> 对比展示与 PNG 下载。
"""
from __future__ import annotations

import os
import time

import numpy as np
import streamlit as st

import scanner

# 点击选点组件（可选依赖，缺失时自动退化为坐标输入方式）
try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_CLICK = True
except Exception:  # pragma: no cover - 取决于运行环境
    HAS_CLICK = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

st.set_page_config(page_title="文档扫描预处理工具", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------

def reset_for_new_file(sig: str) -> None:
    """上传新图片时重置全部处理状态。"""
    st.session_state.update(
        sig=sig,
        img=None,
        scale=1.0,
        det=None,
        warped=None,
        click_points=[],
        last_click=None,
    )
    for key in list(st.session_state.keys()):
        if key.startswith("mpt_"):
            del st.session_state[key]


if "sig" not in st.session_state:
    reset_for_new_file("")


# ---------------------------------------------------------------------------
# 页面头部
# ---------------------------------------------------------------------------

st.title("📄 文档扫描预处理工具")
st.caption(
    "上传文档照片，自动检测纸张四角并完成透视矫正与扫描增强；"
    "自动检测失败时可手动指定四个角点。处理结果可下载为 PNG，适合作为 OCR 前置步骤。"
)

# ---------------------------------------------------------------------------
# 侧边栏：参数设置
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ 处理参数")
    mode_label = st.radio("扫描输出模式", ["灰度扫描", "黑白扫描"], index=1)
    mode = "gray" if mode_label == "灰度扫描" else "bw"
    denoise = st.checkbox("中值滤波去噪", value=True)
    bg_ksize = st.slider("去阴影背景估计核大小", 15, 151, 51, step=2)
    st.divider()
    st.subheader("黑白二值化（仅黑白模式）")
    block_size = st.slider("自适应阈值邻域大小", 11, 101, 41, step=2)
    threshold_c = st.slider("阈值偏移量 C", 2, 30, 10)
    st.divider()
    st.subheader("自动检测参数")
    canny_low = st.slider("Canny 低阈值", 10, 150, 50)
    canny_high = st.slider("Canny 高阈值", 50, 300, 150)
    min_area_ratio = st.slider("最小面积占比", 0.05, 0.6, 0.2, step=0.05)

# ---------------------------------------------------------------------------
# 图片上传
# ---------------------------------------------------------------------------

uploaded = st.file_uploader("上传文档照片（JPG / JPEG / PNG，≤ 20MB）", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("请先上传一张文档照片开始处理。")
    st.stop()

data = uploaded.getvalue()
if len(data) == 0:
    st.error("上传的文件为空，请重新选择图片。")
    st.stop()
if len(data) > scanner.MAX_FILE_BYTES:
    st.error("文件超过 20MB 限制，请压缩后再上传。")
    st.stop()

sig = f"{uploaded.name}:{len(data)}"
if st.session_state.sig != sig:
    reset_for_new_file(sig)

# 解码与尺寸控制（仅在首次上传时执行）
if st.session_state.img is None:
    try:
        raw = scanner.load_image_from_bytes(data)
    except scanner.ScanError as exc:
        st.error(str(exc))
        st.stop()
    img, scale = scanner.resize_to_max(raw, scanner.MAX_IMAGE_DIM)
    st.session_state.img = img
    st.session_state.scale = scale
    st.session_state.det = None

img = st.session_state.img
h, w = img.shape[:2]
if st.session_state.scale < 1.0:
    st.warning(f"图片较大，已等比缩放至 {w}×{h} 进行处理，以保护内存与速度。")

# ---------------------------------------------------------------------------
# 自动检测
# ---------------------------------------------------------------------------

col_btn, col_status = st.columns([1, 3])
run_detect = col_btn.button("🔍 自动扫描文档四角", type="primary")

if run_detect or st.session_state.det is None:
    with st.spinner("正在检测文档轮廓…"):
        st.session_state.det = scanner.detect_document(
            img,
            canny_low=canny_low,
            canny_high=canny_high,
            min_area_ratio=min_area_ratio,
        )
det: scanner.DetectionResult = st.session_state.det

if det.found:
    col_status.success(det.message)
else:
    col_status.warning(det.message)

# 角点来源选择
source_options = ["自动检测", "手动调整"]
default_idx = 0 if det.found else 1
corner_source = st.radio(
    "角点来源", source_options, index=default_idx, horizontal=True,
    help="自动检测失败或结果不正确时，切换到手动调整。",
    key="corner_source_radio",
)

# ---------------------------------------------------------------------------
# 手动四点调整（兜底）
# ---------------------------------------------------------------------------

manual_corners: np.ndarray | None = None
if corner_source == "手动调整":
    with st.expander("🖱️ 手动指定四个角点", expanded=True):
        st.markdown(
            "请按 **左上 → 右上 → 右下 → 左下** 的顺序确定四个角点。"
            + ("可直接在下方图片上点击取点，" if HAS_CLICK else "")
            + "也可以在下方输入框中精确修改坐标。"
        )
        # 初始化默认坐标（向内缩 5%）
        defaults = [
            (w * 0.05, h * 0.05), (w * 0.95, h * 0.05),
            (w * 0.95, h * 0.95), (w * 0.05, h * 0.95),
        ]
        for i in range(4):
            if f"mpt_{i}x" not in st.session_state:
                if det.found:
                    cx, cy = float(det.corners[i][0]), float(det.corners[i][1])
                else:
                    cx, cy = defaults[i]
                st.session_state[f"mpt_{i}x"] = int(round(cx))
                st.session_state[f"mpt_{i}y"] = int(round(cy))

        if HAS_CLICK:
            # 展示当前已选点的叠加预览，点击追加新点
            current = np.array(
                [[st.session_state[f"mpt_{i}x"], st.session_state[f"mpt_{i}y"]] for i in range(4)],
                dtype=np.float32,
            )
            picked = st.session_state.click_points
            overlay_pts = picked if picked else current
            vis = scanner.draw_points(img, overlay_pts)
            disp_w = min(w, 900)
            click = streamlit_image_coordinates(vis, width=disp_w, key="click_img")
            if click is not None:
                xy = (click["x"], click["y"])
                if xy != st.session_state.last_click:
                    st.session_state.last_click = xy
                    px = int(round(xy[0] * w / disp_w))
                    py = int(round(xy[1] * h / (disp_w * h / w)))
                    pts = st.session_state.click_points
                    if len(pts) >= 4:
                        pts = []  # 已集齐四点后再次点击则重新开始
                    pts.append((px, py))
                    st.session_state.click_points = pts
                    for i, (ix, iy) in enumerate(pts):
                        st.session_state[f"mpt_{i}x"] = ix
                        st.session_state[f"mpt_{i}y"] = iy
                    st.rerun()
            cp_cols = st.columns(2)
            cp_cols[0].info(f"已通过点击选择 {len(st.session_state.click_points)}/4 个点")
            if cp_cols[1].button("🔄 重置点击点"):
                st.session_state.click_points = []
                st.session_state.last_click = None
                st.rerun()

        # 坐标微调输入框
        input_cols = st.columns(4)
        labels = ["① 左上", "② 右上", "③ 右下", "④ 左下"]
        for i in range(4):
            with input_cols[i]:
                st.markdown(f"**{labels[i]}**")
                st.number_input("x", 0, w - 1, key=f"mpt_{i}x")
                st.number_input("y", 0, h - 1, key=f"mpt_{i}y")
        manual_corners = np.array(
            [[st.session_state[f"mpt_{i}x"], st.session_state[f"mpt_{i}y"]] for i in range(4)],
            dtype=np.float32,
        )
        st.image(scanner.draw_corners(img, manual_corners), caption="手动角点预览", use_container_width=True)

# ---------------------------------------------------------------------------
# 原图与检测结果对比
# ---------------------------------------------------------------------------

st.subheader("🖼️ 原图与检测预览")
pc1, pc2 = st.columns(2)
pc1.image(scanner.to_rgb(img), caption=f"原始图片（{w}×{h}）", use_container_width=True)
if det.found:
    pc2.image(scanner.draw_corners(img, det.corners), caption="自动检测结果", use_container_width=True)
elif det.edge_image is not None:
    pc2.image(det.edge_image, caption="边缘图（未能筛选出文档四边形）", use_container_width=True)

# ---------------------------------------------------------------------------
# 透视矫正与增强
# ---------------------------------------------------------------------------

st.subheader("🛠️ 透视矫正与扫描增强")
if st.button("✨ 执行透视矫正与增强", type="primary"):
    corners = manual_corners if corner_source == "手动调整" else (det.corners if det.found else None)
    if corners is None:
        st.error("当前没有可用的角点：请先自动检测，或切换到“手动调整”指定四个角点。")
    else:
        try:
            with st.spinner("正在矫正与增强…"):
                st.session_state.warped = scanner.four_point_transform(img, corners)
        except scanner.ScanError as exc:
            st.session_state.warped = None
            st.error(f"透视矫正失败：{exc}")

warped = st.session_state.warped
if warped is not None:
    try:
        enhanced = scanner.enhance(
            warped, mode=mode, denoise=denoise,
            bg_ksize=bg_ksize, block_size=block_size, threshold_c=threshold_c,
        )
    except scanner.ScanError as exc:
        st.error(f"图像增强失败：{exc}")
        st.stop()

    st.subheader("📊 处理结果对比")
    rc1, rc2, rc3 = st.columns(3)
    rc1.image(scanner.to_rgb(img), caption="原始图片", use_container_width=True)
    rc2.image(scanner.to_rgb(warped), caption=f"透视矫正（{warped.shape[1]}×{warped.shape[0]}）", use_container_width=True)
    rc3.image(enhanced, caption=f"{mode_label}结果", use_container_width=True)

    # 下载与本地保存
    st.subheader("💾 结果下载")
    ts = time.strftime("%Y%m%d_%H%M%S")
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ 下载矫正彩色图 (PNG)",
        data=scanner.encode_png(warped),
        file_name=f"corrected_{ts}.png",
        mime="image/png",
    )
    dl2.download_button(
        f"⬇️ 下载{mode_label}扫描图 (PNG)",
        data=scanner.encode_png(enhanced),
        file_name=f"scan_{'gray' if mode == 'gray' else 'bw'}_{ts}.png",
        mime="image/png",
        type="primary",
    )

    with st.expander("保存到本地 output/ 目录（可选）"):
        save_name = st.text_input("文件名（不含扩展名）", value=f"scan_{ts}")
        if st.button("保存到 output/"):
            safe = os.path.basename(save_name).strip() or f"scan_{ts}"
            path = os.path.join(OUTPUT_DIR, f"{safe}.png")
            n = 1
            while os.path.exists(path):  # 避免覆盖已有文件
                path = os.path.join(OUTPUT_DIR, f"{safe}_{n}.png")
                n += 1
            with open(path, "wb") as f:
                f.write(scanner.encode_png(enhanced))
            st.success(f"已保存：{path}")
else:
    st.info("确认角点后，点击上方“执行透视矫正与增强”按钮生成扫描结果。")
