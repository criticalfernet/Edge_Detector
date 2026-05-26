import cv2
import cupy as cp
import numpy as np

from clahe import apply_clahe
from basic_log_cupy import LOG_cupy

def multi_scale_LOG_cupy_frame(src_gray, sigmas, base_t, weight_base_t=0.625):
    src_clahe = apply_clahe(src_gray)
    src_cpu = src_clahe.astype(np.float32)
    src_cupy = cp.asarray(src_cpu)

    responses_gpu = []
    
    for sigma in sigmas:
        edges_cupy = LOG_cupy(src_cupy, sigma, base_t)
        responses_gpu.append(edges_cupy)

    stack = cp.stack(responses_gpu, axis=-1)

    weights = cp.array(sigmas, dtype=cp.float32)
    weighted_sum = cp.sum(stack * weights, axis=-1)

    weight_t = weight_base_t * cp.sum(weights)
    combined = (weighted_sum >= weight_t)
    
    combined_uint8 = (combined * 255).astype(cp.uint8)
    
    return combined_uint8.get()

def multi_scale_LOG_cupy(path, sigmas, base_t, weight_base_t=0.625):
    src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise ValueError(f"Could not read image: {path}")
    
    return multi_scale_LOG_cupy_frame(src, sigmas, base_t, weight_base_t)
