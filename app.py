from __future__ import annotations

import io
from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from vision_core import custom_canny, detect_harris, detect_keypoints, match_two_images, stitch_images, warp_two_images


st.set_page_config(page_title='A3 图像特征检测与匹配', layout='wide')


def read_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法读取图片：{uploaded_file.name}')
    return img


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(title: str, image: np.ndarray, width: str = 'stretch'):
    st.markdown(f'**{title}**')
    st.image(bgr_to_rgb(image), use_container_width=width == 'stretch')


def intro_panel():
    st.title('Vibe Coding 图像特征检测与匹配系统')
    st.caption('支持 Canny 边缘检测、Harris/SIFT 特征点、双图匹配流程可视化，以及多图全景拼接。')
    with st.expander('作业要求对应关系', expanded=False):
        st.markdown(
            '''
            1. **边缘检测**：展示灰度图、梯度幅值、非极大值抑制结果、最终边缘图。  
            2. **特征点检测**：支持 Harris 与 SIFT，并直接在图像上可视化特征点。  
            3. **匹配流程可视化**：展示特征检测、初始匹配、RANSAC 内点匹配、变换后轮廓/对齐。  
            4. **全景拼接**：支持多图上传，比较 Overlay 与 Feather 两种 blending。  
            5. **交互式 Web App**：基于 Streamlit，实现参数滑块、方法切换、结果下载。
            '''
        )


intro_panel()

tab1, tab2, tab3, tab4 = st.tabs(['1 边缘检测', '2 特征点检测', '3 双图匹配流程', '4 多图全景拼接'])

with tab1:
    st.subheader('Canny 边缘检测与 NMS 前后对比')
    file = st.file_uploader('上传一张图片', type=['png', 'jpg', 'jpeg'], key='edge')
    colp1, colp2, colp3 = st.columns(3)
    blur_ksize = colp1.slider('高斯核大小', 3, 11, 5, step=2)
    sigma = colp2.slider('高斯 sigma', 0.5, 3.0, 1.2, 0.1)
    high_ratio = colp3.slider('高阈值比例', 0.10, 0.50, 0.25, 0.01)
    low_ratio = st.slider('低阈值比例', 0.02, 0.30, 0.10, 0.01)

    if file:
        img = read_uploaded_image(file)
        result = custom_canny(img, blur_ksize=blur_ksize, sigma=sigma, low_ratio=low_ratio, high_ratio=high_ratio)
        c1, c2 = st.columns(2)
        with c1:
            show_image('原图', img)
            show_image('梯度幅值（NMS 前）', result.grad_mag)
        with c2:
            show_image('非极大值抑制后', result.nms)
            show_image('最终边缘', result.edges)

with tab2:
    st.subheader('Harris / SIFT 特征点检测')
    file = st.file_uploader('上传一张图片', type=['png', 'jpg', 'jpeg'], key='feat')
    method = st.radio('特征点方法', ['HARRIS', 'SIFT'], horizontal=True)
    if file:
        img = read_uploaded_image(file)
        show_image('原图', img)
        if method == 'HARRIS':
            response, vis = detect_harris(img)
            c1, c2 = st.columns(2)
            with c1:
                show_image('Harris 响应图', response)
            with c2:
                show_image('Harris 特征点可视化', vis)
        else:
            kp, desc, vis = detect_keypoints(img, 'SIFT')
            st.write(f'检测到关键点数量：{len(kp)}')
            show_image('SIFT 特征点可视化', vis)

with tab3:
    st.subheader('两幅图像的特征匹配、RANSAC 与配准')
    f1 = st.file_uploader('上传图像 1', type=['png', 'jpg', 'jpeg'], key='m1')
    f2 = st.file_uploader('上传图像 2', type=['png', 'jpg', 'jpeg'], key='m2')
    colm1, colm2, colm3 = st.columns(3)
    method = colm1.selectbox('检测/描述方法', ['SIFT', 'ORB'])
    ratio = colm2.slider('Lowe ratio', 0.50, 0.95, 0.75, 0.01)
    ransac = colm3.slider('RANSAC 阈值', 1.0, 10.0, 4.0, 0.5)
    blend = st.selectbox('两图融合方式', ['feather', 'overlay'])

    if f1 and f2:
        img1 = read_uploaded_image(f1)
        img2 = read_uploaded_image(f2)
        try:
            res = match_two_images(img1, img2, method=method, ratio_thresh=ratio, ransac_thresh=ransac)
            aligned = warp_two_images(img1, img2, res['H'], blend_method=blend)
            c1, c2 = st.columns(2)
            with c1:
                show_image('初始匹配（ratio test 后）', res['initial_matches_vis'])
                show_image('RANSAC 内点匹配', res['ransac_matches_vis'])
            with c2:
                show_image('单应变换轮廓投影到图像 2', res['outline_vis'])
                show_image('对齐/融合结果', aligned)
            st.write(f'初始匹配数：{len(res["good_matches"])}；RANSAC 内点数：{len(res["inlier_matches"])}')
        except Exception as e:
            st.error(str(e))

with tab4:
    st.subheader('多图全景拼接与 blending 对比')
    files = st.file_uploader('上传 2 张及以上同场景重叠图像', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key='stitch')
    cols1, cols2 = st.columns(2)
    method = cols1.selectbox('拼接特征方法', ['SIFT', 'ORB'], key='stitch_method')
    blend = cols2.selectbox('blending 方法', ['feather', 'overlay'], key='stitch_blend')

    if files and len(files) >= 2:
        imgs: List[np.ndarray] = [read_uploaded_image(f) for f in files]
        st.write(f'已上传 {len(imgs)} 张图像。建议按从左到右或从近到远的顺序上传。')
        preview_cols = st.columns(min(3, len(imgs)))
        for i, img in enumerate(imgs[:3]):
            with preview_cols[i % len(preview_cols)]:
                show_image(f'预览 {i + 1}', img)
        try:
            pano = stitch_images(imgs, method=method, blend_method=blend)
            show_image(f'拼接结果（{blend}）', pano)
            ok, buffer = cv2.imencode('.png', pano)
            if ok:
                st.download_button(
                    '下载拼接结果 PNG',
                    data=buffer.tobytes(),
                    file_name='panorama.png',
                    mime='image/png',
                )
        except Exception as e:
            st.error(f'拼接失败：{e}')
    elif files:
        st.info('请至少上传两张图像。')

st.divider()
st.markdown('**开发说明**：本系统使用 Python + OpenCV + Streamlit 实现；如需写入报告，可在文末注明参考了 ChatGPT / GPT-5.4 进行代码辅助与文档整理。')
