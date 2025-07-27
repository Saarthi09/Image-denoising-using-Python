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