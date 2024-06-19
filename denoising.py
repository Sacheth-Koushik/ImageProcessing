from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from csbdeep.io import save_tiff_imagej_compatible

from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
from PIL import Image
import zipfile


model_name = 'models'
basedir = '/Users/sachethkoushik/Desktop/RCAN processing/base_dir_denoise'
model = N2V(config=None, name=model_name, basedir=basedir)
print(f"model name: {model}")
datagen = N2V_DataGenerator()

imgs1 = datagen.load_imgs_from_directory(directory = "/Users/sachethkoushik/Desktop/RCAN processing/ICIP training data/0/RawDataQA (31)",
                                         filter='*.tiff', dims='YX')


pred = model.predict(imgs1[0][0,:,:,:], axes='YXC')

# Convert the image to grayscale
grayscale_image = pred.convert('L')
grayscale_image.save('grayscale_image.png')

save_tiff_imagej_compatible('pred_train.tiff', grayscale_image, axes='YXC')