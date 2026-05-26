import cupy as cp
from cupyx.scipy import ndimage

def LOG_cupy(src_cupy, alpha, base_t):
    blurred = ndimage.gaussian_filter(src_cupy, sigma=alpha)
    
    laplacian_kernel = cp.array([
        [ 0,  1,  0], 
        [ 1, -4,  1], 
        [ 0,  1,  0]
    ], dtype=cp.float32)
    
    laplacian = ndimage.convolve(blurred, laplacian_kernel)
    laplacian = laplacian * (alpha ** 2)

    threshold_val = base_t * float(cp.mean(cp.abs(laplacian)))
    
    zero_crossings = cp.zeros_like(laplacian, dtype=bool)
    
    shifts = [
        (-1,  0), ( 1,  0),  
        ( 0, -1), ( 0,  1), 
        (-1, -1), (-1,  1), 
        ( 1, -1), ( 1,  1)
    ]
    
    for dy, dx in shifts:
        shifted_map = cp.roll(laplacian, shift=(dy, dx), axis=(0, 1))
        crosses = (laplacian * shifted_map < 0) & (cp.abs(laplacian - shifted_map) > threshold_val)
        zero_crossings = zero_crossings | crosses
        
    return zero_crossings
