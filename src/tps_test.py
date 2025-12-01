import numpy as np
import cv2
import time


def U(r):
    # r shape: (...)
    # Thin plate spline radial basis: r^2 * log(r)
    # Define U(0)=0 to avoid NaN
    r = np.where(r == 0, 1e-8, r)
    return (r * r) * np.log(r)


def tps_kernel_matrix(ctrl):
    """Build the TPS kernel matrix K from control points."""
    n = ctrl.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(ctrl[i] - ctrl[j])
            K[i, j] = U(r)
    return K


def tps_system_matrix(ctrl):
    """Construct the full TPS matrix to solve once."""
    n = ctrl.shape[0]
    K = tps_kernel_matrix(ctrl)

    P = np.hstack([
        np.ones((n, 1)),
        ctrl
    ])

    # Full TPS matrix
    top = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])

    L = np.vstack([top, bottom])  # (n+3) x (n+3)
    return L


def precompute_pixel_basis(h, w, ctrl):
    """
    Precompute the basis U(|x - p_i|) for every pixel and control point.
    Returns basis_x: shape (h, w, n)
    """
    ys, xs = np.indices((h, w), dtype=np.float64)
    pixels = np.stack([xs, ys], axis=-1)  # (h, w, 2)

    n = ctrl.shape[0]
    basis = np.zeros((h, w, n), dtype=np.float64)

    for i in range(n):
        dx = pixels[..., 0] - ctrl[i, 0]
        dy = pixels[..., 1] - ctrl[i, 1]
        r = np.sqrt(dx*dx + dy*dy)
        basis[..., i] = U(r)

    return basis


def get_current_dst(src, t):
    """
    src: (n,2) original control points
    t: time or frame index
    Return: dst with same shape.
    Replace this function with your supplied control points each frame.
    """
    dst = src.copy()
    # example animated deformation: horizontal wobble + small bulge
    dst[:, 0] += 30.0 * np.sin(src[:, 1] * 0.02 + t * 2.0)  # horizontal wobble
    cx, cy = np.mean(src[:, 0]), np.mean(src[:, 1])
    dx = src[:, 0] - cx
    dy = src[:, 1] - cy
    r2 = dx*dx + dy*dy
    dst[:, 0] += 0.0006 * r2 * (1.0 + 0.5*np.sin(t*3.0))
    return dst


# ---------- Precompute basis at mesh vertices ----------
def precompute_vertex_basis(grid_x, grid_y, ctrl):
    """
    grid_x, grid_y: mesh coordinate arrays, shape (gh, gw) with pixel coords
    ctrl: control points (n,2)
    Returns:
      basis: shape (gh, gw, n) of U(|v - p_i|)
    """
    gh, gw = grid_x.shape
    n = ctrl.shape[0]
    basis = np.zeros((gh, gw, n), dtype=np.float64)
    for i in range(n):
        dx = grid_x - ctrl[i, 0]
        dy = grid_y - ctrl[i, 1]
        r = np.sqrt(dx*dx + dy*dy)
        basis[..., i] = U(r)
    return basis

# ---------- Demo hook: replace this with your realtime control points ----------


def get_current_dst(src, t):
    """
    src: (n,2) original control points
    t: time or frame index
    Return: dst with same shape.
    Replace this function with your supplied control points each frame.
    """
    dst = src.copy()
    # example animated deformation: horizontal wobble + small bulge
    dst[:, 0] += 30.0 * np.sin(src[:, 1] * 0.02 + t * 2.0)  # horizontal wobble
    # cx, cy = np.mean(src[:, 0]), np.mean(src[:, 1])
    # dx = src[:, 0] - cx
    # dy = src[:, 1] - cy
    # r2 = dx*dx + dy*dy
    # dst[:, 0] += 0.0006 * r2 * (1.0 + 0.5*np.sin(t*3.0))
    return dst


cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

ret, frame = cam.read()

h, w = frame.shape[:2]

grid_w = 50
grid_h = 50

# Example control points: choose a modest number, e.g. 20 points
# Replace this with your control points: these are the source positions (fixed)
# Here we use a uniform-ish scatter across the face of the frame for demo.
# User must ensure their supplied dst corresponds to these src indices.
ctrl_nx, ctrl_ny = 5, 4
xs = np.linspace(0, w - 1, ctrl_nx)
ys = np.linspace(0, h - 1, ctrl_ny)
grid_cx, grid_cy = np.meshgrid(xs, ys)
src_ctrl = np.vstack([grid_cx.ravel(), grid_cy.ravel()]).T.astype(np.float64)
n_ctrl = src_ctrl.shape[0]

# Build TPS system and inverse once (depends only on src_ctrl)
L = tps_system_matrix(src_ctrl)
# Add tiny regularization for numerical stability
reg = 1e-8 * np.eye(L.shape[0])
L_inv = np.linalg.inv(L + reg)

# Build vertex mesh coordinates (coarse mesh)
vx = np.linspace(0, w - 1, grid_w)
vy = np.linspace(0, h - 1, grid_h)
vgrid_x, vgrid_y = np.meshgrid(vx, vy)  # shape (grid_h, grid_w)
vcoords = np.stack([vgrid_x, vgrid_y], axis=-1)  # (gh, gw, 2)
gh, gw = vgrid_x.shape

# Precompute basis values for vertices (fast)
basis_vertices = precompute_vertex_basis(
    vgrid_x, vgrid_y, src_ctrl)  # (gh, gw, n)

# Flatten basis for fast dot: (gh*gw, n)
basis_flat = basis_vertices.reshape((-1, n_ctrl))  # (V, n)

# Precompute affine part base for vertices: P_v = [1, x, y] for each vertex
ones = np.ones((gh * gw, 1), dtype=np.float64)
vxs = vgrid_x.ravel().reshape(-1, 1).astype(np.float64)
vys = vgrid_y.ravel().reshape(-1, 1).astype(np.float64)
P_v = np.hstack([ones, vxs, vys])  # (V, 3)

# For upsampling, we will form small maps map_x_small (grid_h x grid_w) containing new x for each vertex
# then use cv2.resize to upsample to full resolution (w x h)

previous_time = 0
frame_count = 0
while True:
    ret, frame = cam.read()

    diff_time = time.time() - previous_time
    previous_time = time.time()
    if diff_time > 0:
        fps = 1 / diff_time
    else:
        fps = 0  # FPS is 0 initially or if time delta is too small

    # ---------- USER SUPPLIES dst control points here ----------
    # Replace this call with your live dst array of shape (n_ctrl, 2)
    dst_ctrl = get_current_dst(src_ctrl, time.time())
    # ----------------------------------------------------------

    # Solve TPS small system for X and Y (concat with zeros for affine constraints)
    Vx = np.concatenate([dst_ctrl[:, 0], np.zeros(3)])
    Vy = np.concatenate([dst_ctrl[:, 1], np.zeros(3)])

    params_x = L_inv @ Vx
    params_y = L_inv @ Vy

    w_x = params_x[:n_ctrl]
    a_x = params_x[n_ctrl:]  # a0, ax, ay

    w_y = params_y[:n_ctrl]
    a_y = params_y[n_ctrl:]

    # Evaluate warp at vertices:
    # vertex_x' = basis_flat @ w_x + P_v @ a_x
    vx_prime_flat = basis_flat.dot(w_x) + P_v.dot(a_x)
    vy_prime_flat = basis_flat.dot(w_y) + P_v.dot(a_y)

    # Reshape to (gh, gw)
    map_x_small = vx_prime_flat.reshape((gh, gw)).astype(np.float32)
    map_y_small = vy_prime_flat.reshape((gh, gw)).astype(np.float32)

    # Upsample to full resolution. Note cv2.resize takes (width, height)
    # Use INTER_LINEAR for smoothness, and align corners by default OK for this grid
    map_x_full = cv2.resize(map_x_small, (w, h),
                            interpolation=cv2.INTER_LINEAR)
    map_y_full = cv2.resize(map_y_small, (w, h),
                            interpolation=cv2.INTER_LINEAR)

    # map_x_full/map_y_full now contain for each destination pixel the source x/y.
    # Need to ensure they are float32
    map_x_full = map_x_full.astype(np.float32)
    map_y_full = map_y_full.astype(np.float32)

    warped = cv2.remap(
        frame,
        map_x_full,
        map_y_full,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    fps_text = f"FPS: {int(fps)}"
    cv2.putText(warped, fps_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Camera', warped)
    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
