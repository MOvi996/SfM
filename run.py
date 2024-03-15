from dataloader import Dataset


data = Dataset("Stage_1_Data_ver._3/stage1/box")
data.load_data()

print(data.N_images, data.visibility.shape, data.points.shape, data.colors.shape)