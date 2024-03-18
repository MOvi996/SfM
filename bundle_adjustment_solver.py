from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from utils import *

def is_list_of_lists(obj):
    # Check if the object is a list
    if isinstance(obj, list):
        # Check if every item in the list is also a list
        return all(isinstance(item, list) for item in obj)
    # If obj is not a list, return False
    return False

def flatten_list_of_lists(obj):
    # if is_list_of_lists(obj):
    #     # It's a list of lists, so flatten it
    #     return [item for sublist in obj for item in sublist]
    # else:
    #     # It's not a list of lists, return the original object
    #     return obj
    return [item for sublist in obj for item in sublist]

def bundleAdjustmentLM(X_reconstructed, reconstructed_ind, points_2d, visibility, camera_rotations, camera_translations, K, num_views, appended_ids, max_eval = 3):

    indices = (reconstructed_ind == 1)
    X_3d    = X_reconstructed[indices[:,0]]
    n_points = X_3d.shape[0]
    cam_param_list = []
    for i in range(num_views):
        R, C = camera_rotations[i], camera_translations[i]
        rotvec = Rotation.from_matrix(R).as_rotvec()
        
        if isinstance(C[0], np.ndarray):
            C = C.ravel()
        
        if isinstance(rotvec[0], np.ndarray):
            rotvec = rotvec.ravel()

        RC = np.array([rotvec[0], rotvec[1], rotvec[2], C[0], C[1], C[2]])    
                                                        
        cam_param_list.append(RC[None])


    cam_param_list = np.concatenate(cam_param_list) #.reshape(-1, 6)
    
    x0 = np.hstack((cam_param_list.ravel(), X_3d.ravel()))


    res = least_squares(projectionErrorLM, x0, method= "lm", max_nfev= max_eval, verbose=2, args=(num_views, n_points, indices, visibility,appended_ids, points_2d, K ))
    
    optimized_params = res.x
    optimized_RC = optimized_params[:num_views * 6].reshape((num_views, 6))
    optimized_3d = optimized_params[num_views * 6:].reshape((n_points, 3))

    optimized_X = np.zeros_like(X_reconstructed)
    optimized_X[indices[:,0]] = optimized_3d

    optimized_camera_rotations, optimized_camera_translation = [], []
    for i in range(len(optimized_RC)):
        R = Rotation.from_rotvec(optimized_RC[i, :3]).as_matrix()
        C = optimized_RC[i, 3:].reshape(3,1)
        optimized_camera_translation.append(C)
        optimized_camera_rotations.append(R)

    return optimized_X, optimized_camera_rotations, optimized_camera_translation


def projectionErrorLM(x0, num_views, n_points,indices, camera_visibility, appended_ids, points_2d, K):

    camera_params = x0[:num_views * 6].reshape((num_views, 6))
    points_3d = x0[num_views * 6:].reshape((n_points, 3))
    total_error_vec = []
    for camera in range(num_views):
        R = Rotation.from_rotvec(camera_params[camera,:3]).as_matrix()
        C = camera_params[camera, 3:]
        P = computeProjectionMatrix(K, R, C)
        current_points = points_2d[indices[:,0], appended_ids[camera]]
        current_viz    = camera_visibility[indices[:,0],appended_ids[camera]]

        indices_visible = current_viz == 1
        gt = current_points[indices_visible]
        recons_X = points_3d[indices_visible]
        proj_x = projectPoint(computeProjectionMatrix(K, R, C), recons_X)[:,:2]

        error_vec = (proj_x - gt).ravel()
        total_error_vec.append(error_vec)
    

    return np.concatenate(total_error_vec)

