import numpy as np 
import cv2
import glob 
import os 
from natsort import natsorted
import json 


class Dataset:

    def __init__(self, path_to_directory):

        self.path        = path_to_directory
        self.img_path    = os.path.join(self.path, "images")
        self.corr_path   = os.path.join(self.path, "correspondences")
        self.cam_path    = glob.glob(f"{self.path}/*.json")[0]
        self.images      = []


    def load_data_new(self):
        
        with open(self.cam_path) as json_data:
            camera_parameters = json.load(json_data)
            json_data.close()

        self.K = np.array(camera_parameters['intrinsics'])
        self.correspondence_files = natsorted(glob.glob(f"{self.corr_path}/*"))
        self.image_files          = natsorted(glob.glob(f"{self.img_path}/*"))
        
        self.images_index   = dict()
        self.N_images       = len(self.image_files)
        self.count_graph    = np.zeros((self.N_images, self.N_images))
        self.points         = np.empty((0,self.N_images, 2))
        self.colors         = np.empty((0,3), dtype= np.uint8)
        self.visibility_new = np.empty((0,self.N_images,self.N_images))
        
        for i,image in enumerate(self.image_files):
            
            image_id = int(image.split("/")[-1][:-4])
            self.images_index[image_id] = i
            self.images.append(image_id)
            

        for f in self.correspondence_files:
    
            img1, img2 = [int(x) for x in f.split("/")[-1][:-4].split("_")]
    
            img1_index = self.images_index[img1]
            image_one = cv2.imread(self.image_files[img1_index])
            image_one = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
            
            img2_index = self.images_index[img2]

            data = np.loadtxt(f)
            _, indices = np.unique(data, axis = 0, return_index= True)
            
            data = data[np.sort(indices)]
        
            self.count_graph[img1_index, img2_index] = data.shape[0]
            self.count_graph[img2_index, img1_index] = data.shape[0]
            
            img1_points = np.fliplr(data[:,:2])
            img2_points = np.fliplr(data[:,2:])

            if self.visibility_new.shape[0] != 0:
        
                new_keypoints = []
                new_colors    = []
                new_2d_viz    = []
        
                previous_points = self.points[:,img1_index,:]
                previous_points_2 = self.points[:,img2_index,:]
                
                for p in range(img1_points.shape[0]):

                    index   = (img1_points[p] == previous_points).all(axis = 1).nonzero()[0]
                    index_2 = (img2_points[p] == previous_points_2).all(axis = 1).nonzero()[0]
                    if (not (len(index) > 0)) and  (not (len(index_2) > 0)):
                        
                        v_2d_new = np.zeros((1, self.N_images, self.N_images))
                        v_2d_new[0,img1_index, img2_index] = 1
                        v_2d_new[0,img2_index, img1_index] = 1
                        
                        k_new = np.zeros((1, self.N_images,2))
                        k_new[0,img1_index,:] = img1_points[p]
                        k_new[0,img2_index,:] = img2_points[p]
                
                        new_keypoints.append(k_new)
                        new_colors.append(image_one[int(img1_points[p,0]),int(img1_points[p,1])][None])
                        new_2d_viz.append(v_2d_new)

                    elif len(index) > 0:

                        index = index[0]
                        self.points[index, img2_index, :] = img2_points[p]
                        self.visibility_new[index, img2_index, img1_index] = 1
                        self.visibility_new[index, img1_index, img2_index] = 1

                    else:

                        index_2 = index_2[0]
                        self.points[index_2, img1_index] = img1_points[p]
                        self.visibility_new[index_2, img2_index, img1_index] = 1
                        self.visibility_new[index_2, img1_index, img2_index] = 1

                if len(new_keypoints) > 0:
                    new_keypoints = np.concatenate(new_keypoints)
                    new_colors    = np.concatenate(new_colors)
                    new_2d_viz    = np.concatenate(new_2d_viz)
                    self.points         = np.vstack([self.points, new_keypoints])
                    self.colors         = np.vstack([self.colors, new_colors])    
                    self.visibility_new = np.vstack([self.visibility_new, new_2d_viz])

            else:
                new_keypoints     = np.zeros((data.shape[0], self.N_images, 2))
                new_2d_visibility = np.zeros((data.shape[0], self.N_images, self.N_images))

                new_2d_visibility[:,img1_index, img2_index] = 1
                new_2d_visibility[:,img2_index, img1_index] = 1
                
                new_keypoints[:,img1_index,:] = img1_points 
                new_keypoints[:,img2_index,:] = img2_points

                image_coords = img1_points.astype(np.int32)
                colors_new = image_one[image_coords[:,0],image_coords[:,1]]
        
                self.points         = np.vstack([self.points, new_keypoints])
                self.colors         = np.vstack([self.colors, colors_new])
                self.visibility_new = np.vstack([self.visibility_new, new_2d_visibility])

    def load_data(self):
        
        with open(self.cam_path) as json_data:
            camera_parameters = json.load(json_data)
            json_data.close()

        self.K = np.array(camera_parameters['intrinsics'])
        self.correspondence_files = natsorted(glob.glob(f"{self.corr_path}/*"))
        self.image_files          = natsorted(glob.glob(f"{self.img_path}/*"))
        
        self.images_index = dict()
        self.N_images     = len(self.image_files)
        self.count_graph  = np.zeros((self.N_images, self.N_images))
        self.visibility = np.empty((0,self.N_images))
        self.points     = np.empty((0,self.N_images, 2))
        self.colors     = np.empty((0,3), dtype= np.uint8)

        for i,image in enumerate(self.image_files):
            
            image_id = int(image.split("/")[-1][:-4])
            self.images_index[image_id] = i
            self.images.append(image_id)
            

        for f in self.correspondence_files:
    
            img1, img2 = [int(x) for x in f.split("/")[-1][:-4].split("_")]
    
            img1_index = self.images_index[img1]
            image_one = cv2.imread(self.image_files[img1_index])
            image_one = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
            
            img2_index = self.images_index[img2]

            data = np.loadtxt(f)
            _, indices = np.unique(data, axis = 0, return_index= True)
            
            data = data[np.sort(indices)]
            self.count_graph[img1_index, img2_index] = data.shape[0]
            self.count_graph[img2_index, img1_index] = data.shape[0]
            
            img1_points = np.fliplr(data[:,:2])
            img2_points = np.fliplr(data[:,2:])

            if self.visibility.shape[0] != 0:
        
                new_points    = []
                new_keypoints = []
                new_colors    = []
        
                previous_points = self.points[:,img1_index,:]
                previous_points_2 = self.points[:,img2_index,:]
                
                for p in range(img1_points.shape[0]):

                    index   = (img1_points[p] == previous_points).all(axis = 1).nonzero()[0]
                    index_2 = (img2_points[p] == previous_points_2).all(axis = 1).nonzero()[0]
                    if (not (len(index) > 0)) and  (not (len(index_2) > 0)):
                        
                        v_new = np.zeros((1, self.N_images))
                        v_new[0,img1_index] = 1
                        v_new[0,img2_index] = 1
                        k_new = np.zeros((1, self.N_images,2))
                        k_new[0,img1_index,:] = img1_points[p]
                        k_new[0,img2_index,:] = img2_points[p]
                
                        new_points.append(v_new)
                        new_keypoints.append(k_new)
                        new_colors.append(image_one[int(img1_points[p,0]),int(img1_points[p,1])][None])
            
                    elif len(index) > 0:

                        index = index[0]
                        
                        self.visibility[index, img2_index] = 1
                        self.points[index, img2_index, :] = img2_points[p]
                    else:

                        index_2 = index_2[0]
                        self.visibility[index_2, img1_index] = 1
                        self.points[index_2, img1_index] = img1_points[p]

                if len(new_points) > 0:
                    new_points    = np.concatenate(new_points)
                    new_keypoints = np.concatenate(new_keypoints)
                    new_colors    = np.concatenate(new_colors)

                    self.visibility = np.vstack([self.visibility, new_points])
                    self.points     = np.vstack([self.points, new_keypoints])
                    self.colors     = np.vstack([self.colors, new_colors])    

            else:
                new_keypoints = np.zeros((data.shape[0], self.N_images, 2))
                new_points    = np.zeros((data.shape[0], self.N_images))

                new_points[:,img1_index] = 1
                new_points[:,img2_index] = 1
                new_keypoints[:,img1_index,:] = img1_points 
                new_keypoints[:,img2_index,:] = img2_points

                image_coords = img1_points.astype(np.int32)
                colors_new = image_one[image_coords[:,0],image_coords[:,1]]
        
                self.visibility = np.vstack([self.visibility, new_points])
                self.points     = np.vstack([self.points, new_keypoints])
                self.colors     = np.vstack([self.colors, colors_new])
        
    def computeSceneGraph(self):
        
        images_to_add = []
        appended_ids  = []

        for iteration in range(self.N_images - 1):

            if iteration == 0:
                max_ind = np.array((self.count_graph == self.count_graph.max()).nonzero())

                if max_ind.shape[1] > 2:
                    selected_id = np.argmax(np.abs(max_ind[1,:] - max_ind[0,:]))
                else:
                    selected_id = 0

                id_one = max_ind[0,selected_id]
                id_two = max_ind[1,selected_id]
                images_to_add.append(self.images[id_one])
                images_to_add.append(self.images[id_two])
                appended_ids.append(id_one)
                appended_ids.append(id_two)
                self.count_graph[id_one, id_two] = 0
                self.count_graph[id_two, id_one] = 0
            else:

                sub_graph = self.count_graph[:,appended_ids]
                max_ind = np.array((sub_graph == sub_graph.max()).nonzero())

                if max_ind.shape[1] > 2:
                    selected_id = np.argmax(np.abs(max_ind[1,:] - max_ind[0,:]))
                else:
                    selected_id = 0
            
                id_one = max_ind[0, selected_id]
                id_two = appended_ids[max_ind[1,selected_id]]
            
                images_to_add.append(self.images[id_one])
            
                appended_ids.append(id_one)
            
                for ids in appended_ids[:-1]:

                    self.count_graph[id_one, ids] = 0
                    self.count_graph[ids, id_one] = 0

        return images_to_add, appended_ids