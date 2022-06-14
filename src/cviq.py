# Correlating the performance of computer vision algorithms relative to objective image quality
import numpy as np
from wand.image import Image
import sys
import os

root = './'
reference_source = 'Reference-Images'
augmented_output = 'Augmented-Images'
cwd = os.getcwd()

        
with Image(filename=root+reference_source+'/esfriso.png') as img:
    for blur in range(0,5,1):
        blur_img = img
        blur_img.blur(blur,blur+1)

    
        for noise in np.arange(0.0,1.0,0.2):
            noise = np.around(noise, decimals=1)
            noise_img=blur_img
            noise_img.noise(noise_type='gaussian', attenuate=noise)

            noise_img.save(filename=root+augmented_output+'/output_image_noise' + str(noise) +'_blur' + str(blur) + '.png')

    
