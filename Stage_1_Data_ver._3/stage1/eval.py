import numpy as np
import json
from scipy.spatial import KDTree
import trimesh

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def crop_points_to_bbox(points, bb):
    # Create a boolean mask for points within the bounding box
    mask = np.logical_and(
        np.all(points >= bb[0], axis = 1),
        np.all(points <= bb[1], axis = 1)
    )
    
    # Apply the mask to the points array to get the cropped points
    cropped_points = points[mask]
    
    return cropped_points

def compute_transformation_error(t1, t2):
    eps = 1e-6

    # Rotation
    r1 = t1[:3, :3]
    r2 = t2[:3, :3]
    rot_error = (np.trace(r1 @ r2.T) - 1) / 2
    rot_error = np.clip(rot_error, -1.0 + eps, 1.0 - eps)
    rot_error = np.arccos(rot_error)

    # Translation
    tr1 = t1[:3, 3]
    tr2 = t2[:3, 3]
    tr_error = np.linalg.norm(tr1 - tr2)

    return (rot_error, tr_error)

def pose_estimate(d1, d2, scale):
    total_error_rotation = 0.0
    total_error_translation = 0.0

    keys = d1.keys()

    for camera in keys:
        transform1 = np.array(d1[camera], dtype = np.float32)
        transform2 = np.array(d2[camera], dtype = np.float32)
        transform2[:3, 3] /= scale

        rot_err, tr_err = compute_transformation_error(transform1, transform2)

        total_error_rotation += rot_err
        total_error_translation += tr_err

    total_error_rotation /= len(keys)
    total_error_translation /= len(keys)

    return total_error_rotation, total_error_translation


if __name__ == "__main__":
    eps = 1e-8

    f_gt = open("box/gt_camera_parameters.json", "r")
    f_predicted = open("box/estimated_camera_parameters.json", "r")

    camera_param_gt = json.load(f_gt)
    camera_param_predicted = json.load(f_predicted)

    scale = 0
    cameras = camera_param_gt["extrinsics"].keys()
    for camera in cameras:
        if camera == "00000.jpg":
            continue

        predicted_camera1 = np.array(camera_param_predicted["extrinsics"][camera])
        predicted_camera2 = np.array(camera_param_gt["extrinsics"][camera])

        scale += (np.linalg.norm(predicted_camera1[:3, 3]) / (np.linalg.norm(predicted_camera2[:3, 3] + eps)))
    scale /= (len(cameras) - 1)

    pcd_gt = trimesh.load("box/gt_points.ply")
    gt_points = np.array(pcd_gt.vertices, dtype = np.float32)

    pcd_test = trimesh.load("box/estimated_points.ply") # Give path to your .ply here (as above)
    test_points = np.array(pcd_test.vertices, dtype = np.float32) / scale

    bb = [
        np.array([-4.523355, -1.264923,  0.198537]), 
        np.array([4.976645, 2.235077, 9.698537])
    ] # Only for box evaluation, do not change

    gt_points = crop_points_to_bbox(gt_points, bb)
    test_points = crop_points_to_bbox(test_points, bb)

    rotation_error, translation_error = pose_estimate(camera_param_gt["extrinsics"], camera_param_predicted["extrinsics"], scale)

    print("Rotation error:", round(rotation_error, 2))
    print("Translation error:", round(translation_error, 3))
    print("Chamfer distance between pointclouds:", chamfer_distance(gt_points, test_points))



