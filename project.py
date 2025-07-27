import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
user_input = input("Enter file name: ").strip()

if not os.path.isabs(user_input):
    filename = os.path.join(script_dir, user_input)
else:
    filename = user_input

if not os.path.isfile(filename):
    print("File cannot be opened:", filename)
    quit()