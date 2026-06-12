import numpy as np

def apply_wb(image, r=1, g=1, b=1):
    rgb_vec = np.array([[[r, g, b]]])
    rgb_vec = rgb_vec / rgb_vec.mean()
    image *= rgb_vec
    image = image.clip(0, 1)
    return image

def white_and_black_point(raw, black=0, white=1):
    raw -= black
    slope = 1/(white - black)
    raw *= slope
    return raw.clip(0, 1)


def exposure(raw, exposure=1):
    return (raw * exposure).clip(0, 1)


def saturation(raw, sat=1):
    mean = raw.mean(axis=2, keepdims=True)
    diff = (raw - mean)*sat + mean
    return diff.clip(0, 1)

def apply_gamma_s_curve(image, gamma=0.4, alpha=5.5, beta=0.4, ):
    image = apply_gamma(image, gamma=gamma)
    image = apply_parametric_s_curve(image, alpha=alpha, beta=beta)
    return image

def apply_gamma(image, gamma=0.4):
    image  = image ** gamma
    return image

def apply_parametric_s_curve(image, alpha=3.0, beta=0.5, method='logistic'):
    """
    Applies a parametric S-curve to a post-gamma image normalized between [0, 1].
    
    Parameters:
    -----------
    image : ndarray
        Input image (NumPy array), values must be in the range [0.0, 1.0].
    alpha : float
        Contrast factor. 
        For 'logistic': alpha > 0 increases contrast (0 is identity).
        For 'rational': alpha > 1 increases contrast (1 is identity).
    beta : float
        The pivot/inflection point of the curve, typically between [0.1, 0.9].
        Values near 0.5 target midtones.
    method : str
        'logistic' for the industry standard, 'rational' for a faster algebraic curve.
        
    Returns:
    --------
    ndarray
        The contrast-enhanced image bounded perfectly between [0.0, 1.0].
    """
    img_clipped = np.clip(image, 0.0, 1.0)
    
    if method == 'logistic':
        if alpha <= 0:
            return img_clipped
        
        def _sigmoid(x, a, b):
            return 1.0 / (1.0 + np.exp(-a * (x - b)))
        
        g_x = _sigmoid(img_clipped, alpha, beta)
        g_0 = _sigmoid(0.0, alpha, beta)
        g_1 = _sigmoid(1.0, alpha, beta)
        
        out = (g_x - g_0) / (g_1 - g_0)
        
    elif method == 'rational':
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0 for rational method.")
            
        eps = 1e-7
        img_safe = np.clip(img_clipped, eps, 1.0 - eps)
        
        num = (img_safe ** alpha) * ((1.0 - beta) ** alpha)
        den = num + ((1.0 - img_safe) ** alpha) * (beta ** alpha)
        out = num / den
        
    else:
        raise ValueError("Method must be either 'logistic' or 'rational'.")
        
    return np.clip(out, 0.0, 1.0)


def pipeline(
        raw, 
        gamma = 1/2.2,
        alpha = 5.5,
        beta = 0.4,
        exposure_value = 1,
        white = 1, black = 0,
        r = 1, g=1, b=1,
        sat = 1,
        method = 'logistic',
    ):
    raw = exposure(raw, exposure=exposure_value)
    raw = apply_wb(raw, r, g, b)
    raw = saturation(raw, sat=sat)
    raw = white_and_black_point(raw, black, white )
                                
    raw = apply_gamma(raw, gamma=gamma)
    raw = apply_parametric_s_curve(raw, alpha=alpha, beta=beta, method=method)
    return raw