import cv2
import numpy as np
import os
from corr_utils import find_correspondences
from stage2_config import *


class Correspondence_Extractor:
    def __init__(self, parent_dir, img_dir='images', stage=2):

        self.img_dir = f'{parent_dir}/{img_dir}'
        print(os.path.exists(self.img_dir))
        print(os.listdir(self.img_dir))
        self.img_list = [f'{self.img_dir}/{img}' for img in os.listdir(self.img_dir) if img.endswith('.jpg') or img.endswith('.png')]
        if stage == 3:
            self.img_names = [os.path.splitext(img)[0].split('/')[-1] for img in self.img_list]
        else:
            self.img_names = [int(img.split('/')[-1].split('.')[0]) for img in self.img_list]
        self.img_list.sort()
        self.img_names.sort()
        
        self.target_folder = f'{parent_dir}/{target_folder}'
        self.stage = stage


        # os.makedirs(self.target_folder, exist_ok=True)
    
    def get_corr_pairs(self, start_index, end_index):

        os.makedirs(self.target_folder, exist_ok=True)

        if end_index == -1:
            end_index = len(self.img_list)

        # Match each image with every other image
        
        for i in range(start_index, end_index):
            if self.stage==3:
                j_range = (i + 20) if (i + 20) < len(self.img_list) else len(self.img_list) - i
            else:
                j_range = len(self.img_list)
            for j in range(i + 1, j_range):
                if i < j:
                    print("matching image {} with image {}".format(self.img_names[i],self.img_names[j]))
                    image2 = cv2.imread(self.img_list[j], cv2.IMREAD_GRAYSCALE)
                    image1 = cv2.imread(self.img_list[i], cv2.IMREAD_GRAYSCALE)

                    try:
                        kp1, kp2, matches = find_correspondences(image1, image2, method=approach, filter=filter_matches, fb_consistency=fb_consistency, threshold=THRESHOLD)
                    except Exception as e:
                        print(f'Error in processing {self.img_list[i]} and {self.img_list[j]}: \n {e}')
                        continue
                    
                    if kp1 is None or kp2 is None or matches is None:
                        print(f'No matches found between {self.img_list[i]} and {self.img_list[j]}')
                        continue
                    
                    # save matches in txt file. Overwrite if file already exists
                    with open(f'{self.target_folder}/{self.img_names[i]}_{self.img_names[j]}.txt', 'w') as f:
                        for match in matches:
                                pts1 = np.array(kp1[int(match[0])].pt)
                                pts2 = np.array(kp2[int(match[1])].pt)
                                if FLIP:
                                    pts1 = np.flip(pts1)
                                    pts2 = np.flip(pts2)
                                f.write(f'{pts1[0]} {pts1[1]} {pts2[0]} {pts2[1]}\n')
                    
                    # save distances in txt file
                    with open(f'{self.target_folder}/{self.img_names[i]}_{self.img_names[j]}_distances.txt', 'w') as f:
                        for match in matches:
                            f.write(f'{match[2]}\n')



if __name__ == "__main__":
    ce = Correspondence_Extractor('stage3/', img_dir='rgb', stage=3 )
    ce.get_corr_pairs(start_index, end_index)