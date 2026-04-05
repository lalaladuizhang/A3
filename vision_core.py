from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CannyResult:
    gray: np.ndarray
    blurred: np.ndarray
    grad_mag: np.ndarray
    nms: np.ndarray
    edges: np.ndarray


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(gray: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def sobel_gradients(blurred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    return gx, gy, mag, angle


def non_maximum_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 0.0
            r = 0.0
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif 112.5 <= a < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]
            if mag[i, j] >= q and mag[i, j] >= r:
                out[i, j] = mag[i, j]
    return out


def double_threshold_hysteresis(nms: np.ndarray, low_ratio: float = 0.1, high_ratio: float = 0.25) -> np.ndarray:
    high = nms.max() * high_ratio
    low = high * low_ratio / max(high_ratio, 1e-8)

    strong = 255
    weak = 75
    out = np.zeros_like(nms, dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms >= low) & (nms < high))
    out[strong_i, strong_j] = strong
    out[weak_i, weak_j] = weak

    h, w = out.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if out[i, j] == weak:
                if np.any(out[i - 1 : i + 2, j - 1 : j + 2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


def custom_canny(image: np.ndarray, blur_ksize: int = 5, sigma: float = 1.2, low_ratio: float = 0.1, high_ratio: float = 0.25) -> CannyResult:
    gray = to_gray(image)
    blurred = gaussian_blur(gray, blur_ksize, sigma)
    _, _, mag, angle = sobel_gradients(blurred)
    nms = non_maximum_suppression(mag, angle)
    edges = double_threshold_hysteresis(nms, low_ratio, high_ratio)
    return CannyResult(gray=gray, blurred=blurred, grad_mag=normalize_to_uint8(mag), nms=normalize_to_uint8(nms), edges=edges)


def get_feature_detector(method: str):
    method = method.upper()
    if method == 'SIFT' and hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(nfeatures=800), cv2.NORM_L2
    if method == 'HARRIS':
        return None, None
    return cv2.ORB_create(nfeatures=1200), cv2.NORM_HAMMING


def detect_harris(image: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04, thresh_ratio: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    gray = np.float32(to_gray(image))
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    mask = dst > thresh_ratio * dst.max()
    ys, xs = np.where(mask)
    for x, y in zip(xs, ys):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), 1)
    return normalize_to_uint8(dst), vis


def detect_keypoints(image: np.ndarray, method: str = 'SIFT') -> Tuple[List[cv2.KeyPoint], np.ndarray | None, np.ndarray]:
    if method.upper() == 'HARRIS':
        response, vis = detect_harris(image)
        return [], None, vis
    gray = to_gray(image)
    detector, _ = get_feature_detector(method)
    keypoints, desc = detector.detectAndCompute(gray, None)
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )
    return keypoints, desc, vis


def match_two_images(img1: np.ndarray, img2: np.ndarray, method: str = 'SIFT', ratio_thresh: float = 0.75, ransac_thresh: float = 4.0):
    gray1, gray2 = to_gray(img1), to_gray(img2)
    detector, norm_type = get_feature_detector(method)
    if detector is None:
        raise ValueError('匹配流程请使用 SIFT 或 ORB，不建议使用 Harris 直接做描述子匹配。')
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ValueError('检测到的有效特征点不足，无法完成匹配。')

    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    if len(good) < 4:
        raise ValueError(f'通过 Lowe ratio 后仅剩 {len(good)} 个匹配点，无法估计单应矩阵。')

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise ValueError('RANSAC 未能估计出稳定的单应矩阵。')
    mask = mask.ravel().astype(bool)
    inlier_matches = [m for i, m in enumerate(good) if mask[i]]

    init_vis = cv2.drawMatches(img1, kp1, img2, kp2, good[:80], None, flags=2)
    ransac_vis = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:80], None, flags=2)

    h1, w1 = img1.shape[:2]
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    outline_vis = img2.copy()
    cv2.polylines(outline_vis, [np.int32(warped_corners)], True, (0, 255, 0), 3)

    return {
        'kp1': kp1,
        'kp2': kp2,
        'des1': des1,
        'des2': des2,
        'good_matches': good,
        'inlier_matches': inlier_matches,
        'H': H,
        'initial_matches_vis': init_vis,
        'ransac_matches_vis': ransac_vis,
        'outline_vis': outline_vis,
    }


def warp_two_images(img1: np.ndarray, img2: np.ndarray, H: np.ndarray, blend_method: str = 'feather') -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, H)
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    out_w, out_h = xmax - xmin, ymax - ymin

    warped1 = cv2.warpPerspective(img1, T @ H, (out_w, out_h))
    canvas2 = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas2[ty:ty + h2, tx:tx + w2] = img2

    mask1 = (warped1.sum(axis=2) > 0).astype(np.float32)
    mask2 = (canvas2.sum(axis=2) > 0).astype(np.float32)

    if blend_method.lower() == 'overlay':
        pano = warped1.copy()
        pano[ty:ty + h2, tx:tx + w2] = img2
        return pano

    dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    weights1 = dist1 / (dist1 + dist2 + 1e-6)
    weights2 = dist2 / (dist1 + dist2 + 1e-6)
    weights1 = weights1[..., None]
    weights2 = weights2[..., None]
    pano = warped1.astype(np.float32) * weights1 + canvas2.astype(np.float32) * weights2
    only1 = (mask1 > 0) & (mask2 == 0)
    only2 = (mask2 > 0) & (mask1 == 0)
    pano[only1] = warped1[only1]
    pano[only2] = canvas2[only2]
    return np.clip(pano, 0, 255).astype(np.uint8)


def stitch_images(images: Sequence[np.ndarray], method: str = 'SIFT', blend_method: str = 'feather') -> np.ndarray:
    if len(images) < 2:
        raise ValueError('至少需要两张图像才能拼接。')
    pano = images[0]
    for nxt in images[1:]:
        result = match_two_images(pano, nxt, method=method)
        pano = warp_two_images(pano, nxt, result['H'], blend_method=blend_method)
    return pano
