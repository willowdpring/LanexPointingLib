import os
import PIL
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.transforms as trans
from matplotlib.patches import Polygon
import matplotlib.ticker as tick
from cv2 import (
    getPerspectiveTransform,
    perspectiveTransform,
    warpPerspective,
    fillConvexPoly,
    INTER_LINEAR
)
from pointing2d_settings import settings
import pointing2d_fit as fit
import pointing2d_lib
from scipy.interpolate import griddata


def vprint(message, keep=False):
    if settings.verbose:
        if keep:
            print(message,end="")
        else:
            print(message)

def ceil2(num): # rounding up to a power of 2
    return np.pow(np.ceil(np.log2(num)), 2)

def cart2sph(x, y, z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq))  # theta
    az = m.atan2(y, x)  # phi
    return r, elev, az
    
def src_dst_from_known_points(known_points, units, resolution, lanex_onAx_dist, lanex_theta, lanex_inPlane_dist, lanex_height, lanex_vertical_offset):
    """
    generates src, dst arrays for the perspective transform

    Parameters
    ----------
    known_points :
        tuple(list[4](tuple)) ie [[[px1,py1], ...][[Lx1,Ly1], ...] with lanex coords in mm from center
        the pixel src and lanex dst in mm to be converted to theta phi
      OR
        dict [int,int,int]
            A 4 element dict with each corner eg 'TR' being a list of [rulermark,p_x,p_y]

    units: int
        units/radian (typically 1000)
    resolution: int
        pixels/unit (typically 10)
    lanex_onAx_dist: [float]
        mm distance to the lanex plane in the axis of the laser
    lanex_theta: [float]
        deg angle of the normal of the lanex plane to the laser
        # WARN: this assumes that the lanex plane is vertical and only rotates about z
    lanex_inPlane_dist: [float]
        mm distance of the edge of the lanex (0mm ruler mark)
        from the axis in the plane of the lanex -ve implies that the laser axes intersects the lanex
    lanex_height: [float]
        mm height of the lanex from the plane of the laser
    lanex_vertical_offset: [float]
        mm height of the center of the lanex wrt. the laser level

    Returns
    -------
    src : np.array[,float32]
        list of source points in pixel x,y
    dst : np.array[,float32]
        list of destination points in theta phi [rad/units]

    """
    if len(known_points) == 2 and len(known_points[0]) == 4:
        vprint("pixel and mm coordinates provided in full:")
        return known_points_from_PX_LX(
            known_points,
            units,
            resolution,
            lanex_onAx_dist,
            lanex_theta,
            lanex_inPlane_dist,
            lanex_height,
            lanex_vertical_offset,
        )
    else:
        vprint("assuming ruler marks on lanex, height determined by .lanex_height ")
        return known_points_from_ruler_marks(
            known_points,
            units,
            resolution,
            lanex_onAx_dist,
            lanex_theta,
            lanex_inPlane_dist,
            lanex_height,
            lanex_vertical_offset,
        )

    return None
    
def known_points_from_PX_LX(known_points,units,resolution,lanex_onAx_dist,lanex_theta,lanex_inPlane_dist,lanex_height,lanex_vertical_offset):
    """
    generates src, dst arrays for the perspective transform

    Parameters
    ----------
    see src_dst_from_known_points
    """
    src = []
    dst = []

    cosT = np.cos(np.radians(lanex_theta))
    sinT = np.sin(np.radians(lanex_theta))

    centroid = np.mean(known_points[1], axis=0) - [settings.dx, settings.dh]

    # Step 2: Calculate the angle of each point relative to the centroid
    def angle_from_centroid(point):
        # Subtract the centroid to translate the points
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        # Return the angle of the point relative to the x-axis
        return np.arctan2(dy, dx)

    angles = [angle_from_centroid(px) for px in known_points[1]]
    sorted_indices = np.argsort(angles)

    # Step 3: Sort points based on the angle
    sorted_px = [known_points[0][i] for i in sorted_indices]
    sorted_lx = [np.subtract(known_points[1][i], centroid) for i in sorted_indices]

    for i, PX in enumerate(sorted_px):
        LX = sorted_lx[i]
        vprint("mapping {} to {}".format(PX, LX))
        src.append(PX)

        x = -cosT * (lanex_inPlane_dist + LX[0])
        z = lanex_onAx_dist - sinT * (lanex_inPlane_dist + LX[0])

        y = LX[1]

        r, p, t = cart2sph(z, x, y)

        vprint("x:{}, y:{}, z:{}".format(x, y, z))
        vprint("r:{}, theta:{}, phi:{}".format(r, t, p))
        dst.append([units * resolution * t, units * resolution * p])

    vprint("src: {}\ndst: {}".format(src, dst))

    return(np.array(src), np.array(dst))


def known_points_from_ruler_marks(known_points,units,resolution,lanex_onAx_dist,lanex_theta,lanex_inPlane_dist,lanex_height,lanex_vertical_offset):
    """
    generates src, dst arrays for the perspective transform

    Parameters
    ----------
    see src_dst_from_known_points
    """
    if settings.verbose:
        print("Generating transformation coords from known points: ")

    order = ["TR", "BR", "BL", "TL"]

    src = []
    dst = []

    cosT = np.cos(np.radians(lanex_theta))
    sinT = np.sin(np.radians(lanex_theta))

    for corner in order:
        vprint("{} Corner".format(corner))
        src.append([known_points[corner][1], known_points[corner][2]])

        x = -cosT * (lanex_inPlane_dist + known_points[corner][0])
        z = lanex_onAx_dist - sinT * (lanex_inPlane_dist + known_points[corner][0])

        if "T" in corner:
            y = lanex_vertical_offset + (0.5 * lanex_height)
        else:
            y = lanex_vertical_offset - (0.5 * lanex_height)
        r, p, t = cart2sph(z, x, y)
        vprint("x:{}, y:{}, z:{}".format(x, y, z))
        vprint("r:{}, theta:{}, phi:{}".format(r, t, p))
        dst.append([units * resolution * t, units * resolution * p])
    vprint("src: {}\ndst: {}".format(src, dst))

    return (np.array(src), np.array(dst))


def jacobian_det_grid(H, dst_h, dst_w):
    """
    Compute |det J| at every dst pixel by mapping back to src via H_inv,
    then evaluating the analytical Jacobian of H at those src points.
    Returns shape (dst_h, dst_w).
    """
    H_inv = np.linalg.inv(H)
    ys, xs = np.mgrid[0:dst_h, 0:dst_w].astype(np.float64)
    ones    = np.ones_like(xs)
    dst_pts = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=0)  # 3 x N

    src_pts = H_inv @ dst_pts
    src_pts /= src_pts[2]                        # dehomogenise
    sx, sy = src_pts[0], src_pts[1]

    # Evaluate Jacobian of H (not H_inv) at those src coords
    w     = H[2,0]*sx + H[2,1]*sy + H[2,2]
    num_u = H[0,0]*sx + H[0,1]*sy + H[0,2]
    num_v = H[1,0]*sx + H[1,1]*sy + H[1,2]

    du_dx = (H[0,0]*w - num_u*H[2,0]) / w**2
    du_dy = (H[0,1]*w - num_u*H[2,1]) / w**2
    dv_dx = (H[1,0]*w - num_v*H[2,0]) / w**2
    dv_dy = (H[1,1]*w - num_v*H[2,1]) / w**2

    jac = np.abs(du_dx * dv_dy - du_dy * dv_dx).reshape(dst_h, dst_w)
    return jac


def get_transform(src, dst, dst_offset):
    """
    Generate or load a cached perspective transform that maps src points
    to dst points, with the coordinate origin translated by dst_offset.

    Parameters
    ----------
    src : array-like, shape (4, 2), float32
        Source points in pixel coordinates.
    dst : array-like, shape (4, 2), float32
        Destination points (e.g. theta/phi).
    dst_offset : tuple (cx, cy)
        Pixel offset of the beam axis in the destination canvas.

    Returns
    -------
    H_eff : np.ndarray, shape (3, 3), float64
        Effective perspective transform including the origin translation.
    """
    if settings.transformation is not None:
        if settings.verbose:
            print("Loading cached transform ... ", end="")
        H_eff = settings.transformation
        if settings.verbose:
            print("Done")
    else:
        if settings.verbose:
            print("Calculating transform ... ", end="")

        cx, cy = dst_offset
        T = np.array([[1, 0, cx],
                      [0, 1, cy],
                      [0, 0,  1]], dtype=np.float64)

        H = getPerspectiveTransform(
            np.array(src, dtype=np.float32),
            np.array(dst, dtype=np.float32)
        )
        H_eff = T @ H.astype(np.float64)

        settings.transformation = H_eff

        if settings.verbose:
            print("Done")

    return H_eff



def transformPoints(points_xy, warp_transform):
    pts   = np.array(points_xy, dtype=np.float32)
    ones  = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    pts_t = (warp_transform @ pts_h.T).T
    w     = pts_t[:, 2:3]
    return (pts_t[:, :2] / w).astype(np.float32)


    
def get_jacobian_weights(H, dst_h, dst_w):
    """
    Generate or load cached Jacobian-based intensity correction weights
    for a given transform H over a (dst_h, dst_w) canvas.

    The Jacobian determinant of H at each destination pixel measures the
    local area stretching: jac > 1 means src area was compressed into
    fewer dst pixels (so we brighten), jac < 1 means it was spread over
    more (so we dim).

    Parameters
    ----------
    H : np.ndarray, shape (3, 3), float64
        Perspective transform (as returned by get_transform).
    dst_h : int
    dst_w : int

    Returns
    -------
    weight_map : np.ndarray, shape (dst_h, dst_w), float32
        Per-pixel weights = 1 / jac, zeroed where jac is near zero.
    jac : np.ndarray, shape (dst_h, dst_w), float32
        Raw Jacobian determinant map.
    """
    if settings.transformation_weights is not None:
        if settings.verbose:
            print("Loading cached Jacobian weights ... ", end="")
        weight_map, jac = settings.transformation_weights
        if settings.verbose:
            print("Done")
    else:
        if settings.verbose:
            print("Calculating Jacobian weights ... ", end="")

        jac        = jacobian_det_grid(H, dst_h, dst_w)
        weight_map = np.where(jac > 1e-8, 1.0 / jac, 0.0).astype(np.float32)

        settings.transformation_weights = (weight_map, jac)

        if settings.verbose:
            print("Done")

    return weight_map, jac


def integral_preserving_warp(src_img, dst_size, dst_offset, src, dst):
    """
    Warp src_img by a perspective transform defined by src->dst point
    correspondences, placing the coordinate origin at dst_offset, with
    Jacobian correction to preserve the image integral.

    Parameters
    ----------
    src_img : 2D np.ndarray
        Input image.
    dst_size : tuple (dst_w, dst_h)
        Output canvas size in pixels.
    dst_offset : tuple (cx, cy)
        Pixel coordinates of the beam axis (origin) in the dst canvas.
    src : array-like, shape (4, 2)
        Source reference points in pixel coordinates.
    dst : array-like, shape (4, 2)
        Destination reference points (e.g. theta/phi).

    Returns
    -------
    dst_corrected : np.ndarray, float32, shape (dst_h, dst_w)
        Warped and intensity-corrected image.
    jac : np.ndarray, float32, shape (dst_h, dst_w)
        Jacobian determinant map.
    axis : tuple (cx, cy)
        Beam axis position in dst — i.e. dst_offset, echoed for convenience.
    """
    dst_w, dst_h = dst_size
    cx, cy       = dst_offset

    H          = get_transform(src, dst, dst_offset)
    warped     = warpPerspective(src_img, H, (int(dst_w), int(dst_h)),
                                     flags=INTER_LINEAR)
    weight_map, jac = get_jacobian_weights(H, dst_h, dst_w)

    dst_corrected = (warped * weight_map).astype(np.float32)

    return dst_corrected, (cx, cy), jac

def get_dst_layout(src_shape, src, dst, pad, resolution):
    """
    Compute or load cached destination canvas dimensions and beam-axis offset.

    The canvas is sized to contain the full warped source image, with `pad`
    pixels of margin. The origin offset (ox, oy) places the physical origin
    (0, 0) in dst-space at the correct canvas pixel.

    Parameters
    ----------
    src_shape  : tuple (H, W) — shape of the source image
    src        : array-like, shape (N, 2) — source points in pixel coords
    dst        : array-like, shape (N, 2) — dest points in physical units
    pad        : tuple (pad_x, pad_y) — canvas padding in pixels
    resolution : float — pixels per physical unit

    Returns
    -------
    dst_size   : tuple (dst_w, dst_h) — canvas size in pixels
    dst_offset : tuple (ox, oy)       — beam axis position in canvas pixels
    """
    if settings.dst_layout is not None:
        if settings.verbose:
            print("Loading cached dst layout ... ", end="")
        dst_size, dst_offset = settings.dst_layout
        if settings.verbose:
            print("Done")
    else:
        if settings.verbose:
            print("Calculating dst layout ... ", end="")

        src_h, src_w = src_shape

        src_pts    = np.array(src, dtype=np.float32)
        dst_pts_px = np.array(dst, dtype=np.float32) * resolution
        H          = getPerspectiveTransform(src_pts[:4], dst_pts_px[:4])

        src_corners = np.array(
            [[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32
        )
        dst_corners = transformPoints(src_corners, H)

        min_x, min_y = dst_corners.min(axis=0)
        max_x, max_y = dst_corners.max(axis=0)

        ox     = int(pad[0]-min_x)
        oy     = int(pad[1]-min_y)
        dst_w  = int((max_x - min_x) + 2 * pad[0])
        dst_h  = int((max_y - min_y) + 2 * pad[1])

        dst_size   = (dst_w, dst_h)
        dst_offset = (ox, oy)

        settings.dst_layout = (dst_size, dst_offset)

        if settings.verbose:
            print("Done")

    return dst_size, dst_offset


def check_transformation(
    pixelData,
    src,
    dst,
    pad,
    units,
    resolution,
    zoom_radius,
    saveDir=None,
):
    """
    Parameters
    ----------
    pixelData   : np.ndarray  (H x W) or (H x W x C), float32 or uint8
    src         : np.ndarray, shape (N, 2), float32 — pixel (x, y) coords
    dst         : np.ndarray, shape (N, 2), float32 — physical coords in
                  'units' (e.g. mrad), origin-centred
    pad         : tuple (pad_x, pad_y) — canvas padding in pixels
    units       : float — units/radian (e.g. 1000 for mrad)
    resolution  : float — pixels/unit
    zoom_radius : float — units from origin to show in zoom panel
    saveDir     : str or None
    """

    # ── Normalise input image to float32 grayscale ───────────────────────────
    img = np.array(pixelData, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.max() > 1.0:
        img = img / 255.0

    # ── Canvas dimensions and beam axis ──────────────────────────────────────
    dst_size, dst_offset = get_dst_layout(img.shape, src, dst, pad, resolution)
    dst_w, dst_h         = dst_size
    ox, oy               = dst_offset

    # ── Apply warp with Jacobian correction ───────────────────────────────────
    transformed, axis, jac = integral_preserving_warp(
        img, dst_size, dst_offset, src, dst
    )

    # ── Integral conservation check ───────────────────────────────────────────
    src_sum = float(img.sum())
    dst_sum = float(transformed.sum())
    print(f"Source integral      : {src_sum:.4f}")
    print(f"Destination integral : {dst_sum:.4f}")
    print(f"Ratio (should be ~1) : {dst_sum/src_sum:.6f}")

    # ── Reconstruct H for the inverse overlay in panel 0 ─────────────────────
    src_pts    = np.array(src, dtype=np.float32)
    dst_pts_px = np.array(dst, dtype=np.float32) * resolution
    H          = getPerspectiveTransform(src_pts[:4], dst_pts_px[:4])
    H_inv      = np.linalg.inv(H)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(18, 8), width_ratios=(1.8, 2, 1.4))

    x_tick_every = 5 * int(zoom_radius / 25)
    y_tick_every = 5 * int(zoom_radius / 25)

    # --- full transformed panel ticks ---
    xticks, xlabels = [], []
    for i in range(transformed.shape[1]):
        if abs(np.floor(i - axis[0])) % (x_tick_every * resolution) == 0:
            lab = int(np.floor((i - axis[0]) / resolution))
            xticks.append(i)
            xlabels.append("" if lab in (-10, 0) else str(lab))

    yticks, ylabels = [], []
    for i in range(transformed.shape[0]):
        if abs(np.floor(i - axis[1])) % (y_tick_every * resolution) == 0:
            lab = int(np.floor((i - axis[1]) / resolution))
            ylabels.append("" if lab == 10 else str(-lab))
            yticks.append(i)

    # Panel 0 — raw source
    ax[0].imshow(pixelData, cmap="gray")
    ax[0].set_title("Raw Data")
    origin_dst = np.array([[ox, oy]], dtype=np.float32)
    origin_src = transformPoints(origin_dst, H_inv)
    ax[0].plot(*origin_src[0], "bx", ms=14, mew=2, label="mapped origin")
    ax[0].legend(fontsize=8)

    # Panel 1 — full transformed + Jacobian overlay
    ax[1].imshow(transformed, cmap="gray")
    jac_log = np.where(jac > 1e-8, np.log10(jac), np.nan)
    jac_im  = ax[1].imshow(jac_log, cmap="RdBu", alpha=0.35, vmin=-1, vmax=1)
    fig.colorbar(jac_im, ax=ax[1], label="log₁₀(|J|)  blue=compress  red=stretch")
    ax[1].set_title("Transformed to Theta-Phi\n(integral preserving)")
    ax[1].spines["left"].set_position(("data", axis[0]))
    ax[1].spines["bottom"].set_position(("data", axis[1] + 1))
    ax[1].set_xticks(xticks, labels=xlabels, rotation=90)
    ax[1].set_yticks(yticks, labels=ylabels)
    ax[1].set_xlabel(r"$\theta \quad \times10^{{{}}}$rad".format(int(-np.log10(units))))
    ax[1].set_ylabel(r"$\phi  \quad \times10^{{{}}}$rad".format(int(-np.log10(units))))

    # Panel 2 — zoomed, colour scaled to zoomed region only
    zoom_px     = zoom_radius * resolution
    zoom_x_lims = [max(int(axis[0] - zoom_px), 0),
                   min(int(axis[0] + zoom_px), dst_w)]
    zoom_y_lims = [min(int(axis[1] + zoom_px), dst_h),
                   max(int(axis[1] - zoom_px), 0)]

    z_xticks, z_xlabels = [], []
    for i in range(transformed.shape[1]):
        if abs(np.floor(i - axis[0])) % (x_tick_every * (resolution / 2)) == 0:
            lab = int(np.floor((i - axis[0]) / resolution))
            z_xticks.append(i)
            z_xlabels.append("" if lab in (-5, 0) else str(lab))

    z_yticks, z_ylabels = [], []
    for i in range(transformed.shape[0]):
        if abs(np.floor(i - axis[1])) % (y_tick_every * (resolution / 2)) == 0:
            lab = int(np.floor((i - axis[1]) / resolution))
            z_yticks.append(i)
            z_ylabels.append("" if lab == 5 else str(-lab))

    zoom_crop = transformed[zoom_y_lims[1]:zoom_y_lims[0],
                            zoom_x_lims[0]:zoom_x_lims[1]]
    nonzero   = zoom_crop[zoom_crop > 0]
    dvmin     = float(np.nanmin(nonzero)) if nonzero.size else 0.0
    dvmax     = float(np.nanmax(zoom_crop))

    ax[2].imshow(transformed, vmin=dvmin, vmax=dvmax)
    ax[2].autoscale(False)
    ax[2].set_title(r"Zoomed $\pm{}\times10^{{{}}}$rad".format(
        zoom_radius, int(-np.log10(units))))
    ax[2].spines["left"].set_position(("data", axis[0]))
    ax[2].spines["bottom"].set_position(("data", axis[1] + 1))
    ax[2].set_xticks(z_xticks, labels=z_xlabels, rotation=90)
    ax[2].set_yticks(z_yticks, labels=z_ylabels)
    ax[2].set_ylim(*zoom_y_lims)
    ax[2].set_xlim(*zoom_x_lims)
    ax[2].set_xlabel(r"$\theta \quad \times10^{{{}}}$rad".format(int(-np.log10(units))))
    ax[2].set_ylabel(r"$\phi  \quad \times10^{{{}}}$rad".format(int(-np.log10(units))))

    fig.suptitle(
        f"Integral conservation: src={src_sum:.1f}  dst={dst_sum:.1f}  "
        f"ratio={dst_sum/src_sum:.4f}",
        fontsize=11
    )
    fig.tight_layout()
    plt.show()

    return fig, ax, transformed, jac

def main():
    settings.update_user_settings(None)
    src, dst = src_dst_from_known_points(settings.known_points, 
                                        settings.units, 
                                        settings.resolution, 
                                        settings.lanex_onAx_dist, 
                                        settings.lanex_theta, 
                                        settings.lanex_inPlane_dist, 
                                        settings.lanex_height, 
                                        settings.lanex_vertical_offset)
    
    pixelData = np.array(PIL.Image.open(settings.pointingCalibrationImage))
    
    check_transformation(
        pixelData,
        src,
        dst,
        settings.dst_padding,
        settings.units,
        settings.resolution,
        settings.zoom_radius,
        saveDir=None,
    )

if __name__ == "__main__":
    main()