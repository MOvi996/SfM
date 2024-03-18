FLIP = False
start_index = 1
end_index = -1
approach = 'SIFT'
filter_matches = True
fb_consistency = True
THRESHOLD = 250

# file postfixes
f = "RANSAC" if filter_matches else 'NoFilter'
c = "FB" if fb_consistency else 'NoFB'
fl = "FLIP" if FLIP else 'NoFLIP'

target_folder = f"correspondences_stage_3_{approach}_{f}_{c}_{fl}"
