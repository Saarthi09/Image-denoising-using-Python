import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) #this will set the path for the user and the input file's location will be in reference to this path
user_input = input("Enter file name: ").strip()

if not os.path.isabs(user_input):
    filename = os.path.join(script_dir, user_input) #in case the file isnt available 
else:
    filename = user_input

if not os.path.isfile(filename): #this will close the program in case of file failure (failure of acquirin)
    print("File cannot be opened:", filename)
    quit()

#now we will use the input image and do the work required
#we will start by opening, reading and converting the image to grayscale
image = Image.open(filename).convert('L')
#now convert the image to array 
img_array = np.asarray(image)

#defining PCA
def pca_denoise(image, num_components, axis=0):
    if axis == 1: #PCA now operates across coloumns instead of rows
        image = image.T #Transpose

    pca = PCA(n_components=min(num_components, image.shape[1])) #applying pca
    transformed = pca.fit_transform(image) #merging both fit + transform 
    reconstructed = pca.inverse_transform(transformed)
    
    if axis == 1:
        reconstructed = reconstructed.T #transposes back to normal 

    return np.clip(reconstructed, 0, 255) #makes sure that pixel values lie from 0 to 255 

#now we will use our defined function
denoised_img = pca_denoise(img_array, num_components=50)

#the input image is now denoised so we will use matplotlib to display the changes
#first we display the original image in grayscale
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title("Before")
plt.imshow(img_array, cmap='gray') #this will display in grayscale 
plt.axis("off")

plt.subplot(1,2,2)
plt.title("After")
plt.imshow(denoised_img, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
