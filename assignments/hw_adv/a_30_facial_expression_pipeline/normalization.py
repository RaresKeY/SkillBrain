import numpy as np

def normalize_landmarks(landmarks, width, height):
    """
    Normalizes MediaPipe face landmarks to a canonical centered, rotated, and scaled frame.
    
    Args:
        landmarks: List of MediaPipe landmark objects (with .x, .y, .z attributes).
        width: Image width in pixels.
        height: Image height in pixels.
        
    Returns:
        np.array: Flattened array of transformed coordinates [x0, y0, z0, x1, y1, z1, ...].
    """
    
    # Scale to pixels and flip Y (so Up is +)
    # MediaPipe: x [0..1], y [0..1] (down), z (scaled to x)
    pts = np.array([[lm.x * width, (1.0 - lm.y) * height, lm.z * width] for lm in landmarks])

    # 1. Center on Nose Tip (Index 1)
    nose = pts[1]
    pts -= nose

    # 2. Rotate to align (Rigid Approach)
    # Define target axes using rigid landmarks (Eyes and Forehead/Nose)
    # Right (+X): Left Eye Outer (33) -> Right Eye Outer (263)
    
    v_right_raw = pts[263] - pts[33]
    v_x = v_right_raw / np.linalg.norm(v_right_raw)
    
    # Use Forehead (10) - Nose (1) as a reference for Up/Back vector
    v_ref_up = pts[10] - pts[1]
    
    # Forward (+Z) is perpendicular to Right (+X) and the Ref Up vector
    v_z = np.cross(v_x, v_ref_up)
    v_z /= np.linalg.norm(v_z)
    
    # Up (+Y) is perpendicular to Z and X
    v_y = np.cross(v_z, v_x)
    v_y /= np.linalg.norm(v_y)

    # Rotation Matrix R = [Right, Up, Fwd]
    R = np.vstack([v_x, v_y, v_z]) # 3x3
    
    # Transform: P_new = P @ R.T
    pts_transformed = pts @ R.T
    
    # 3. Normalize Size (Rigid Scaling)
    # Scale based on Inter-Ocular Distance (33 to 263)
    iod = np.linalg.norm(pts_transformed[263] - pts_transformed[33])
    
    if iod > 0:
        target_iod = 0.3
        pts_transformed *= (target_iod / iod)
        
    return pts_transformed.flatten()
