import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.path import Path

import cv2
# from shapely.geometry import Polygon, Point
# from shapely.prepared import prep 


def build_convex_polygon(ref_points):
    hull = ConvexHull(ref_points)
    hull_polygon = ref_points[hull.vertices]
    delaunay = Delaunay(hull_polygon)
    return hull_polygon, hull.vertices, delaunay


def build_nonconvex_polygon_with_ref(ref_points, ref_point, k=5):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(ref_points)
    dists, indices = nbrs.kneighbors([ref_point])
    neighbors = ref_points[indices[0][1:]]

    vecs = neighbors - ref_point
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_neighbors = neighbors[sorted_indices]

    polygon = np.vstack([ref_point, sorted_neighbors, ref_point])
    delaunay = Delaunay(polygon)
    return polygon, delaunay


def is_inside_polygon(point, polygon):
    path = Path(polygon)
    return path.contains_point(point)


def sample_cal_point_around_ref(ref_point, min_dist, max_dist):
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(min_dist, max_dist)
    new_point = ref_point + dist * np.array([np.cos(angle), np.sin(angle)])
    return new_point, dist


def rotate_around_center(point, center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vec = point - center
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    rotated_vec = R @ vec
    return center + rotated_vec



def sample_cal_point_with_rotation_constraint(ref_point, polygon, min_dist, max_dist, rotate_angle_deg=10, max_attempts=100, must_in=False):
    for _ in range(max_attempts):
        candidate, dist = sample_cal_point_around_ref(ref_point, min_dist, max_dist)
        inside = is_inside_polygon(candidate, polygon)

        rotated_cw = rotate_around_center(candidate, ref_point, -rotate_angle_deg)
        inside_cw = is_inside_polygon(rotated_cw, polygon)

        rotated_ccw = rotate_around_center(candidate, ref_point, rotate_angle_deg)
        inside_ccw = is_inside_polygon(rotated_ccw, polygon)

        if inside == inside_cw == inside_ccw:
            # print('Cal', inside, inside_cw, inside_ccw)
            if (not must_in) or inside:
                # print('ref dist', dist)
                return candidate, inside, dist
    return None, None, None


def sample_opposite_point_with_angle_range(ref_point, cal_point, min_dist, max_dist, max_angle_deg=30):
    vec = cal_point - ref_point
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.array([1.0, 0.0])
        norm = 1.0
    base_dir = -vec / norm
    angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))

    def rotate_vector(v, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ v

    rotated_dir = rotate_vector(base_dir, angle_rad)
    dist = np.random.uniform(min_dist, max_dist)
    opp_point = ref_point + dist * rotated_dir
    return opp_point

def sample_opposite_point_with_angle_range_both_rot(ref_point, cal_point, polygon, min_dist, max_dist, max_angle_deg=30, rotate_angle_deg=10, max_attempts=10, inside_cal=None):
    vec = cal_point - ref_point
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.array([1.0, 0.0])
        norm = 1.0
    base_dir = -vec / norm
    

    def rotate_vector(v, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ v
    
    for _ in range(max_attempts):
        angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        rotated_dir = rotate_vector(base_dir, angle_rad)
        dist = np.random.uniform(min_dist, max_dist)
        opp_point = ref_point + dist * rotated_dir


        inside = is_inside_polygon(opp_point, polygon)

        rotated_cw = rotate_around_center(opp_point, ref_point, -rotate_angle_deg)
        inside_cw = is_inside_polygon(rotated_cw, polygon)

        rotated_ccw = rotate_around_center(opp_point, ref_point, rotate_angle_deg)
        inside_ccw = is_inside_polygon(rotated_ccw, polygon)
        if inside == inside_cw == inside_ccw == (not inside_cal):
            # print('opp dist', dist)
            # print('OPP', inside, inside_cw, inside_ccw)
            return opp_point, inside, dist
    
    return None, None, None


def get_polygon_pixel_coords_via_mask(points, delta=50):
    points_yx = np.array(points, dtype=np.int32) + delta
    polygon_points = points_yx[:, ::-1] 
    # polygon_points = np.array(points, dtype=np.int32) 
    
    mask = np.zeros((500, 500), dtype=np.uint8)
    
    cv2.fillPoly(mask, [polygon_points], 1)
    
    y_coords, x_coords = np.where(mask == 1)
    
    all_coords_yx = np.stack([y_coords, x_coords], axis=1) - delta
    
    return all_coords_yx

# [1/2 dist, 2 dist] -> opp
def sample_opposite_point_with_angle_range_both_rot_multi(ref_point, cal_point, polygon, min_dist, max_dist, max_angle_deg=30, rotate_angle_deg=10, max_attempts=10, inside_cal=None):
    vec = cal_point - ref_point
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.array([1.0, 0.0])
        norm = 1.0
    base_dir = -vec / norm
    

    def rotate_vector(v, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ v
    
    for _ in range(max_attempts):
        angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        rotated_dir = rotate_vector(base_dir, angle_rad)
        dist = np.random.uniform(min_dist, max_dist)
        opp_point = ref_point + dist * rotated_dir


        inside = is_inside_polygon(opp_point, polygon)

        rotated_cw = rotate_around_center(opp_point, ref_point, -rotate_angle_deg)
        inside_cw = is_inside_polygon(rotated_cw, polygon)

        rotated_ccw = rotate_around_center(opp_point, ref_point, rotate_angle_deg)
        inside_ccw = is_inside_polygon(rotated_ccw, polygon)
        if inside == inside_cw == inside_ccw == (not inside_cal):
            sample_angle_deg = 25
            opp_point_close = ref_point + min_dist * rotated_dir
            opp_point_far = ref_point + (max_dist + 5) * rotated_dir
            opp_close_rot1 = rotate_around_center(opp_point_close, ref_point, -sample_angle_deg)
            opp_close_rot2 = rotate_around_center(opp_point_close, ref_point, sample_angle_deg)
            opp_far_rot1 = rotate_around_center(opp_point_far, ref_point, -sample_angle_deg)
            opp_far_rot2 = rotate_around_center(opp_point_far, ref_point, sample_angle_deg)
            polygon_points = [opp_point_close, opp_close_rot1, opp_far_rot1, opp_point_far, opp_far_rot2, opp_close_rot2]
            opp_points = get_polygon_pixel_coords_via_mask(polygon_points)

            # print('opp dist', dist)
            # print('OPP', inside, inside_cw, inside_ccw)
            # print(opp_point, opp_points)
            return opp_points, inside, dist
    
    return None, None, None


def get_near_points(ref_points, ref_idx, min_dist, max_dist, rot_ang, ref_must_in):
    ref_point = ref_points[ref_idx]
    rotate_angle_deg = rot_ang

    k = min(5, ref_points.shape[0]-1 )
    polygon, delaunay = build_nonconvex_polygon_with_ref(ref_points, ref_point, k=k)

    cal_point, inside_cal = sample_cal_point_with_rotation_constraint(
        ref_point, polygon, min_dist, max_dist, rotate_angle_deg, max_attempts=200, must_in=ref_must_in)

    if cal_point is None:
        return None

    opp_point = sample_opposite_point_with_angle_range(ref_point, cal_point, min_dist, max_dist, max_angle_deg=30)
    inside_opp = is_inside_polygon(opp_point, polygon)
    
    rotated_cw = rotate_around_center(cal_point, ref_point, -rotate_angle_deg)
    rotated_ccw = rotate_around_center(cal_point, ref_point, rotate_angle_deg)

    return cal_point, opp_point, rotated_cw, rotated_ccw, inside_cal, inside_opp


def get_near_points_both_rot(ref_points, ref_p, min_dist, max_dist, rot_ang, ref_must_in):
    ref_point = ref_p
    rotate_angle_deg = rot_ang

    k = min(5, ref_points.shape[0]-1 )
    polygon, delaunay = build_nonconvex_polygon_with_ref(ref_points, ref_point, k=k)

    cal_point, inside_cal, ref_dist = sample_cal_point_with_rotation_constraint(
        ref_point, polygon, min_dist, max_dist, rotate_angle_deg, max_attempts=200, must_in=ref_must_in)
    
    # sample_opposite_point_with_angle_range_both_rot(ref_point, cal_point, polygon, min_dist, max_dist, max_angle_deg=30, rotate_angle_deg=10, max_attempts=10)
    # opp_point = sample_opposite_point_with_angle_range(ref_point, cal_point, min_dist, max_dist, max_angle_deg=30)
    # inside_opp = is_inside_polygon(opp_point, polygon)

    opp_point, inside_opp, opp_dist = sample_opposite_point_with_angle_range_both_rot(
        ref_point, cal_point, polygon, min_dist, max_dist, max_angle_deg=30, rotate_angle_deg=rotate_angle_deg, max_attempts=10, inside_cal=inside_cal)
    
    if (cal_point is None) or (opp_point is None):
        # print('cal_opp', cal_point, opp_point)
        return None
    
    print('ref/opp dist', ref_dist, opp_dist)
    rotated_cw = rotate_around_center(cal_point, ref_point, -rotate_angle_deg)
    rotated_ccw = rotate_around_center(cal_point, ref_point, rotate_angle_deg)

    return cal_point, opp_point, rotated_cw, rotated_ccw, inside_cal, inside_opp


# multi opp
def get_near_points_both_rot_multi_opp(ref_points, ref_p, min_dist, max_dist, rot_ang, ref_must_in):
    ref_point = ref_p
    rotate_angle_deg = rot_ang

    k = min(5, ref_points.shape[0]-1 )
    polygon, delaunay = build_nonconvex_polygon_with_ref(ref_points, ref_point, k=k)

    cal_point, inside_cal, ref_dist = sample_cal_point_with_rotation_constraint(
        ref_point, polygon, min_dist, max_dist, rotate_angle_deg, max_attempts=200, must_in=ref_must_in)

    
    # sample_opposite_point_with_angle_range_both_rot(ref_point, cal_point, polygon, min_dist, max_dist, max_angle_deg=30, rotate_angle_deg=10, max_attempts=10)
    # opp_point = sample_opposite_point_with_angle_range(ref_point, cal_point, min_dist, max_dist, max_angle_deg=30)
    # inside_opp = is_inside_polygon(opp_point, polygon)

    opp_points, inside_opp, opp_dist = sample_opposite_point_with_angle_range_both_rot_multi(
        ref_point, cal_point, polygon, min_dist, max_dist, max_angle_deg=30, rotate_angle_deg=rotate_angle_deg, max_attempts=10, inside_cal=inside_cal)
    
    if (cal_point is None) or (opp_points is None):
        # print('cal_opp', cal_point, opp_point)
        return None
    
    # print('ref/opp dist', ref_dist, opp_dist)
    rotated_cw = rotate_around_center(cal_point, ref_point, -rotate_angle_deg)
    rotated_ccw = rotate_around_center(cal_point, ref_point, rotate_angle_deg)

    return cal_point, opp_points, rotated_cw, rotated_ccw, inside_cal, inside_opp, ref_dist




def main():
    ref_points = np.array([
        [0, 0],
        [5, 0],
        [5, 4],
        [2, 6],
        [0, 4],
        [1.5, 2]
    ])
    
    ref_points = np.array([
        [1, 1],
        [30, 2],
        [10, 20],
        [22, 12],
        [35, 30]
    ])
    i = 3
    ref_point = ref_points[i]

    # min_dist, max_dist = 0.5, 2.0
    min_dist, max_dist = 2, 4

    rotate_angle_deg = 30

    k = min(5, ref_points.shape[0]-1 )
    polygon, delaunay = build_nonconvex_polygon_with_ref(ref_points, ref_point, k=k)

    cal_point, inside_cal = sample_cal_point_with_rotation_constraint(
        ref_point, polygon, min_dist, max_dist, rotate_angle_deg, max_attempts=200)

    if cal_point is None:
        print("无法找到满足旋转约束的 cal_point")
        return
    print(f"Cal point: {cal_point}, inside polygon: {inside_cal}")

    opp_point = sample_opposite_point_with_angle_range(ref_point, cal_point, min_dist, max_dist, max_angle_deg=30)
    inside_opp = is_inside_polygon(opp_point, polygon)
    print(f"Opposite point: {opp_point}, inside polygon: {inside_opp}")
    """
    # sample_cal_point_with_rotation_constraint
    candidate = sample_cal_point_around_ref(ref_point, min_dist, max_dist)
    inside = is_inside_polygon(candidate, polygon)

    rotated_cw = rotate_around_center(candidate, ref_point, -rotate_angle_deg)
    inside_cw = is_inside_polygon(rotated_cw, polygon)

    rotated_ccw = rotate_around_center(candidate, ref_point, rotate_angle_deg)
    inside_ccw = is_inside_polygon(rotated_ccw, polygon)
    if inside == inside_cw == inside_ccw:
        return candidate, inside
    """

    plt.figure(figsize=(40, 40)) # （8， 8）
    plt.plot(ref_points[:, 0], ref_points[:, 1], 'bo', label='Ref points')
    plt.plot(polygon[:, 0], polygon[:, 1], 'k--', label='Non-convex polygon')

    plt.plot(ref_point[0], ref_point[1], 'ms', markersize=12, label='Selected ref point')
    plt.plot(cal_point[0], cal_point[1], 'ro' if inside_cal else 'rx', markersize=10,
             label='Cal point (inside)' if inside_cal else 'Cal point (outside)')
    plt.plot(opp_point[0], opp_point[1], 'go' if inside_opp else 'gx', markersize=10,
             label='Opposite point (inside)' if inside_opp else 'Opposite point (outside)')

    rotated_cw = rotate_around_center(cal_point, ref_point, -rotate_angle_deg)
    rotated_ccw = rotate_around_center(cal_point, ref_point, rotate_angle_deg)
    plt.plot(rotated_cw[0], rotated_cw[1], 'r^', label='Cal point rotated CW')
    plt.plot(rotated_ccw[0], rotated_ccw[1], 'r^', label='Cal point rotated CCW')

    plt.legend()
    plt.axis('equal')
    plt.title("Sampling cal_point and opposite_point using non-convex region")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"sampling_nonconvex_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
