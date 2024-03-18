from dataloader import *
from utils import * 
from bundle_adjustment_solver import bundleAdjustmentLM
import  struct
from stage2_config import *
from stage2 import Correspondence_Extractor
import json

class SFM:

    def __init__(self, config_path):
        
        with open(config_path) as json_data:
            self.config = json.load(json_data)
            json_data.close()
        
        self.datapath = self.config['datapath']

        if self.config["computeCorrespondence"]:
            self.correspondence_extractor = Correspondence_Extractor(self.datapath,
                                                                      target_folder=self.config["corr_folder"],
                                                                      stage=self.config["stage"])
            
            self.correspondence_extractor.get_corr_pairs(start_index, end_index, min_matches=50)

        
        self.data     = Dataset(self.datapath, corr_dir=self.config["corr_folder"])
        self.data.load_data()
    
        self.camera_visibility = self.data.visibility_new.any(axis = 1).astype(np.float64)

        if self.config["sceneGraph"]:

            self.images_to_add, self.appended_ids = self.data.computeSceneGraph()
        else:
            self.images_to_add = self.data.images
            self.appended_ids = np.arange(self.data.N_images)
    
    def run(self):
        ### Running for two views initially
        indices    = self.data.visibility_new[:,self.appended_ids[0], self.appended_ids[1]] == 1
        pts1, pts2 = self.data.points[indices, self.appended_ids[0]], self.data.points[indices, self.appended_ids[1]]
        E = estimateEssentialFromFundamental(estimate_fundamental_matrix(pts1, pts2), self.data.K)
        rotations, translations = decomposeEssentialMat(E)
        points_3d = []
        for i in range(4):
            P1 = computeProjectionMatrix(self.data.K, np.identity(3), np.zeros((3,)))
            P2 = computeProjectionMatrix(self.data.K, rotations[i], translations[i])
            points_3d.append(linearTriangulation(pts1, pts2, P1, P2))

        X,R,C = cheiralityCondition(points_3d, rotations, translations)

        X_reconstructed   = np.zeros((self.data.points.shape[0], 3))
        reconstructed_ind = np.zeros((self.data.points.shape[0], 1))

        X_reconstructed[indices] = X
        reconstructed_ind[indices] = 1

        camera_rotations    = []
        camera_translations = []
        projection_matrix_extrinsic = []

        camera_rotations.append(np.eye(3))
        camera_rotations.append(R)

        camera_translations.append(np.zeros((3,)))
        camera_translations.append(C)

        projection_matrix_extrinsic.append(projectionMatrix4x4(computeProjectionMatrix(np.identity(3), camera_rotations[0], camera_translations[0])))
        projection_matrix_extrinsic.append(projectionMatrix4x4(computeProjectionMatrix(np.identity(3), camera_rotations[1], camera_translations[1])))

        for next_image in range(2, self.data.N_images):
            visibility_points = self.data.visibility_new[:, self.appended_ids[next_image - 1], self.appended_ids[next_image]]
            indices           = visibility_points == 1
            pts1, pts2 = self.data.points[:, self.appended_ids[next_image -1]], self.data.points[:, self.appended_ids[next_image]]
            E = estimateEssentialFromFundamental(estimate_fundamental_matrix(pts1[indices], pts2[indices]), self.data.K)
            rs, ts = decomposeEssentialMat(E)
            indices_new = np.logical_and((1 - reconstructed_ind)[:,0] , visibility_points)
            points_3d = []
            for i in range(4):
                P1 = computeProjectionMatrix(self.data.K, np.identity(3), np.zeros((3,)))
                P2 = computeProjectionMatrix(self.data.K, rs[i], ts[i])
                points_3d.append(linearTriangulation(pts1[indices_new], pts2[indices_new], P1, P2))

            X,R,C = cheiralityCondition(points_3d, rs, ts)

            X_reconstructed[indices_new] = X
            reconstructed_ind[indices_new] = 1
    
            relative_projection = projectionMatrix4x4(computeProjectionMatrix(np.identity(3),R,C))
            absolute_RC = np.matmul(relative_projection, projection_matrix_extrinsic[next_image -1])
            camera_rotations.append(absolute_RC[:3,:3])
            camera_translations.append(np.matmul(- np.linalg.inv(absolute_RC[:3,:3]), absolute_RC[:-1,3].reshape(3,1)))
            projection_matrix_extrinsic.append(absolute_RC)
            
            for previous_images in range(next_image -1):
    
                visibility_points = self.data.visibility_new[:, self.appended_ids[previous_images], self.appended_ids[next_image]]
                indices_new = np.logical_and((1 - reconstructed_ind)[:,0] , visibility_points)
                pts1, pts2 = self.data.points[:, self.appended_ids[previous_images]], self.data.points[:, self.appended_ids[next_image]]
                P1 = computeProjectionMatrix(self.data.K, camera_rotations[previous_images], camera_translations[previous_images])
                P2 = computeProjectionMatrix(self.data.K, camera_rotations[next_image], camera_translations[next_image])
                points_3d = linearTriangulation(pts1[indices_new], pts2[indices_new], P1, P2)
                X_reconstructed[indices_new]   = points_3d
                reconstructed_ind[indices_new] = 1


            if next_image == self.data.N_images - 1:
                max_eval = 3
                X_reconstructed,camera_rotations,camera_translations = bundleAdjustmentLM(X_reconstructed,reconstructed_ind,
                                                                                self.data.points,self.camera_visibility,camera_rotations,
                                                                                camera_translations, self.data.K, next_image +1, self.appended_ids, max_eval= max_eval)

        
            projection_matrix_extrinsic = []
            I_p = projectionMatrix4x4(computeProjectionMatrix(np.identity(3), camera_rotations[0], camera_translations[0]))
        
            projection_matrix_extrinsic.append(I_p)
            projection_matrix_extrinsic.append(projectionMatrix4x4(computeProjectionMatrix(np.identity(3), camera_rotations[1], camera_translations[1])))

            for view in range(2, len(camera_rotations)):
                current = projectionMatrix4x4(computeProjectionMatrix(np.identity(3), camera_rotations[view], camera_translations[view]))
                projection_matrix_extrinsic.append(np.matmul(current, projection_matrix_extrinsic[view -1]))

        X = X_reconstructed[(reconstructed_ind == 1)[...,0]]
        C = self.data.colors[(reconstructed_ind == 1)[...,0]]
        self.write_pointcloud(os.path.join(self.datapath, "predicted_points.ply"),X,C)

        camera_params = {}
        camera_params["extrinsics"] = {}
        camera_params["intrinsics"] = self.data.K.tolist()

        for image_id in range(len(self.images_to_add)):

            key = str(self.images_to_add[image_id]).zfill(5) + ".jpg"
            camera_params["extrinsics"][key] = projection_matrix_extrinsic[image_id].tolist()

        with open(os.path.join(self.datapath, "predicted_camera_parameters.json"), "w") as outfile: 
            json.dump(camera_params, outfile)

    def write_pointcloud(self,filename,xyz_points,rgb_points=None):

        """ creates a .pkl file of the point clouds generated
        """

        assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
        if rgb_points is None:
            rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
        assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

        # Write header of .ply file
        fid = open(filename,'wb')
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                            rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                            rgb_points[i,2].tostring())))
        fid.close()

