import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) #this will set the path for the user
user_input = input("Enter file name: ").strip()

if not os.path.isabs(user_input):
    filename = os.path.join(script_dir, user_input) #in case the file isnt available 
else:
    filename = user_input

if not os.path.isfile(filename): #these will close the program in case of file failure (failure of acquirin)
    print("File cannot be opened:", filename)
    quit()

#now we will use the input image and do the work required
#we will start by opening, reading and converting the image to grayscale
image = Image.open(filename).convert('L')
#now convert the image to array 
img_array = np.asarray(image)

#defining PCA
def pca_denoise(image, num_components, axis=0):
    if axis == 1:
        image = image.T

    pca = PCA(n_components=min(num_components, image.shape[1]))
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)
    
    if axis == 1:
        reconstructed = reconstructed.T

    return np.clip(reconstructed, 0, 255)