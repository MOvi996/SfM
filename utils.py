import cv2
import numpy as np
import os 
import glob

def to_homogeneous(points_2d):

    return np.hstack([points_2d, np.ones((points_2d.shape[0],1))])



def compute_normalization_transform_2D(homogeneous_2D):

    centroid   = np.mean(homogeneous_2D[:,:2], axis = 0)
    scale_orig = ((homogeneous_2D[:,:2] - centroid[None])**2).sum(axis = 1).mean()
    scale = np.sqrt(2 / scale_orig)

    transformation = np.array([[scale,0, -(scale*centroid[0])],[0, scale, -(scale*centroid[1])], [0,0,1]])
    return transformation


def compute_normalization_transform_3D(homogeneous_3D):

    centroid, scale = np.mean(homogeneous_3D[:,:3], axis = 0), np.std(homogeneous_3D[:,:3])
    scale = np.sqrt(3) / scale

    transformation = np.array([[scale,0, 0, -(scale*centroid[0])],[0, scale,0, -(scale*centroid[1])], [0,0,scale, -(scale*centroid[2])], [0,0,0,1]])
    return transformation


def apply_K_inv(K_inv, homogenous_points):
    return np.dot(K_inv, homogenous_points.T).T

def normalize_points(transform, homogenous_points):
    return np.dot(transform, homogenous_points.T).T

def reprojection_error(x,X_homo, K,R,C):

    P = computeProjectionMatrix(K,R,C)

    proj    = np.matmul(P, X_homo.T).T
    proj    = proj / proj[:,2][...,None]
    proj_2d = proj[:,:2]
    error = np.linalg.norm((x - proj_2d), axis =1)
    return error.mean()

def DLT_PnP(points_2d, points_3d, K):

    points_2d_homogeneous = to_homogeneous(points_2d)
    points_3d_homogeneous = to_homogeneous(points_3d)
    K_inv = np.linalg.inv(K)
    points_2d_camera = apply_K_inv(K_inv, points_2d_homogeneous)

    N_points = points_2d_homogeneous.shape[0]
    if N_points == 6:
        num_iters = 1
    else:
        num_iters = 2000
    
    best_R = None
    best_C = None
    best_criterion = np.inf
    for i in range(num_iters):
        
        points = np.random.choice(N_points,6, replace= False)
        camera_2d     = points_2d_camera[points]
        homogenous_3d = points_3d_homogeneous[points]

        A = np.zeros((12, 12))
        for p in range(6):
            A[2*p] = np.concatenate([ np.zeros(4,), -homogenous_3d[p] , camera_2d[p,1]*homogenous_3d[p]])
            A[(2*p) + 1] = np.concatenate([homogenous_3d[p],np.zeros(4,) , -camera_2d[p,0]*homogenous_3d[p]])
            
        U,S,V = np.linalg.svd(A)
        P = V[-1].reshape((3, 4))
        R = P[:, :3]
        u, s, v = np.linalg.svd(R) # to enforce Orthonormality
        R = u @ v

        C = P[:, 3].reshape((-1,1))
        C = np.matmul(- np.linalg.inv(R),C)
        if np.linalg.det(R) < 0:
            R = -R
            C = -C
        
        error = reprojection_error(points_2d, points_3d_homogeneous, K,R,C)

        if error < best_criterion:
            best_criterion = error
            best_C = C
            best_R = R

    return best_R, best_C
            
        


    
    
def estimate_fundamental_matrix(x_prime, x):


    x_prime_homogeneous = to_homogeneous(x_prime)
    x_homogeneous       = to_homogeneous(x)

    transform_x_prime = compute_normalization_transform_2D(x_prime_homogeneous)
    transform_x       = compute_normalization_transform_2D(x_homogeneous)

    x_prime_camera_norm = normalize_points(transform_x_prime, x_prime_homogeneous)
    x_camera_norm = normalize_points(transform_x, x_homogeneous)
    N_points = x_prime.shape[0]
    A = np.zeros((N_points, 9))

    for p in range(N_points):
        A[p,:] = np.kron(x_camera_norm[p], x_prime_camera_norm[p])

    U,S,Vh = np.linalg.svd(A)
    F_norm = Vh[-1].reshape(3,3)
    Uf, Sf, Vf = np.linalg.svd(F_norm)
    Sf[-1] = 0
    F_norm_corrected = Uf @ np.diag(Sf) @ Vf

    F = np.matmul(transform_x.T, np.matmul(F_norm_corrected, transform_x_prime))

    return F / F[2,2]    

def estimateEssentialFromFundamental(fundamental, k):

    E = np.matmul(k.T, np.matmul(fundamental,k))
    U,S,V = np.linalg.svd(E)
    E_estimated = U @ np.diag([1,1,0]) @ V

    return E_estimated / E_estimated[2,2]

def estimate_essential_matrix(pts1, pts2, K):

    N_points = pts1.shape[0]
    if N_points == 8:
        num_iters = 1
    else:
        num_iters = 2000
    
    best_matrix = None
    best_criterion = 0

    K_inv = np.linalg.inv(K)
    pts1_homogeneous = to_homogeneous(pts1)
    pts2_homogeneous = to_homogeneous(pts2)    
    pts1_camera = apply_K_inv(K_inv, pts1_homogeneous)
    pts2_camera = apply_K_inv(K_inv, pts2_homogeneous)    
  
    for i in range(num_iters):
        points = np.random.choice(N_points,8, replace= False)
        
        x_prime_camera = pts1_camera[points]
        x_camera       = pts2_camera[points]

        A = np.zeros((8, 9))

        for p in range(8):
            A[p,:] = np.kron(x_camera[p], x_prime_camera[p])
    
        U,S,Vh = np.linalg.svd(A)
        E_norm = Vh[-1].reshape(3,3)

        Ue, Se, Ve = np.linalg.svd(E_norm)
        Se = [1, 1, 0]

        E = Ue @ np.diag(Se) @ Ve
        criterion = np.sum(np.abs((pts2_camera.T * (E @ pts1_camera.T)).sum(axis = 0)) < 0.001)
        if criterion > best_criterion:
            best_criterion = criterion
            best_matrix = E
    
    return best_matrix, best_criterion

def extract_cam_pose(E, K):
    """
    function to obtain Rotation and Translation
    :param essential_matrix: a 3x3 matrix
    :param w_matrix: a 3x3 matrix that is multiplied with essential matrix
    :return: Rotation and translation matrices
    """
    
    U, S, V_T = np.linalg.svd(E)

    R = []
    C = []

    # rotation matrices
    R.append(np.dot(U, np.dot(np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]]), V_T)))
    R.append(np.dot(U, np.dot(np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]]), V_T)))
    R.append(np.dot(U, np.dot(np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]]).T, V_T)))
    R.append(np.dot(U, np.dot(np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]]).T, V_T)))

    #  translation matrices
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

def decomposeEssentialMat(E):

    U,S,V = np.linalg.svd(E)

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    t1, t2 = U[:,2], -U[:,2]
    R1 = np.matmul(U, np.matmul(W,V))
    R2 = np.matmul(U, np.matmul(W.T, V))

    # Four possible decompositions
    rotations = []
    translations = []

    if np.linalg.det(R1) < 0:

        rotations.append(-R1)
        rotations.append(-R1)

        translations.append(-t1)
        translations.append(-t2)
    else:
        rotations.append(R1)
        rotations.append(R1)

        translations.append(t1)
        translations.append(t2)

    if np.linalg.det(R2) < 0:

        rotations.append(-R2)
        rotations.append(-R2)

        translations.append(-t1)
        translations.append(-t2)
    else:
        rotations.append(R2)
        rotations.append(R2)

        translations.append(t1)
        translations.append(t2)

    return rotations, translations

def computeProjectionMatrix(K,R,C):
    
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))

    return P

def skew_symmetric_matrix(vector):

    if vector.ndim == 1:

        return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])

    skew_matrix = np.zeros((vector.shape[0], 3 ,3 ))
    skew_matrix[:,0,1] = -vector[:,2]
    skew_matrix[:,0,2] =  vector[:,1]
    skew_matrix[:,1,0] =  vector[:,2]
    skew_matrix[:,1,2] = -vector[:,0]
    skew_matrix[:,2,0] = -vector[:,1]
    skew_matrix[:,2,1] =  vector[:,0]
    
    return skew_matrix


def linearTriangulation(pts1, pts2, projection_matrix1, projection_matrix2):

    x_prime = to_homogeneous(pts1)
    x       = to_homogeneous(pts2)
    cross_prime = skew_symmetric_matrix(x_prime)
    cross_x     = skew_symmetric_matrix(x)
    
    constraint_x_prime = np.einsum("bij,jk->bik", cross_prime, projection_matrix1)
    constraint_x       = np.einsum("bij,jk->bik", cross_x, projection_matrix2)
    constraints = np.concatenate([constraint_x_prime[:,:2,:], constraint_x[:,:2,:]], axis = 1)
    U,S,V = np.linalg.svd(constraints)
    points_3d = V[:,-1] / V[:,-1,-1][...,None]
    
    return points_3d[:,:3]

def cheiralityCondition(points_3d, rotations, translations):

    best_criterion = 0
    for i in range(4):
        
        n = 0
        r3 = rotations[i][2,:]
        t  = translations[i]
        for p in range(points_3d[i].shape[0]):
            
            displacement = points_3d[i][p] - t
            if (np.dot(r3, displacement) > 0) and points_3d[i][p,2] > 0:
                n = n + 1

        if n > best_criterion:
            C = translations[i]
            R = rotations[i]
            X = points_3d[i]
            best_criterion = n

    return X, R, C