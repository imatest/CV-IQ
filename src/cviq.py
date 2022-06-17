# Correlating the performance of computer vision algorithms relative to objective image quality
# This example creates a set of augmented images that can be analyzed for how well they can be used for image 
import numpy as np                     # https://numpy.org/
from wand.image import Image           # Wand imagemagic python https://docs.wand-py.org/en/0.6.7/
import matplotlib.pyplot as plt
import cv2                             # OpenCV-python https://pypi.org/project/opencv-python/
import pytesseract                     # OCR engine https://pypi.org/project/pytesseract/
from imatest.it import ImatestLibrary  # Imatest IT https://www.imatest.com/docs/imatest-it-instructions/#Python for instructions on Imatest IT
import sys
import subprocess                      # for calling ImageMagick
import os
import json

imatestLib = ImatestLibrary()
root = '.'
reference_source = 'Reference-Images'
augmented_output = 'Augmented-Images'
output_prefix = 'output_image'
esfriso_reference = 'esfriso.png'

# Main Function
def main():
    prepare_montage()
    image_simulation()


# Montage of images containing all target types, using imagemagick's montage command line tool
# Install imagemagick here: https://imagemagick.org/index.php
# would have used wand but this issue exists: https://github.com/emcconville/wand/issues/575

montage_reference = 'montage.png'
def prepare_montage():
    images = [facial_reference, esfriso_reference, text_reference, qrcode_reference]
    full_path = [reference_source+os.sep+image for image in images]
    montage_output_file = augmented_output + os.sep  +montage_reference
    montage_command = "montage -tile 2x2 -geometry 100% " + " ".join(full_path) + " " +montage_output_file
    subprocess.call(montage_command)

    # Verify the output
    if os.path.exists(montage_output_file):
        montage_image = cv2.imread(montage_output_file)
        print("Montage image \"" + montage_output_file + "\" generated")
    else:
        raise Exception("Montage command failed to generate file")
    return


# Image simulation using wand-python
# See https://wand-py.org/

def image_simulation():
    image_file = os.path.join(root,augmented_output,montage_reference)

    # Iterate through different levels of blur
    for blur in np.arange(0.0,7.0,1.0):
        img = Image(filename=image_file)
        blur_img = img
        blur_img.blur(radius=0,sigma=blur)

        # Iterate through different levels of noise
        for noise in np.arange(0.0,1.4,0.2):
            noise = np.around(noise, decimals=1)
            noise_img=blur_img
            noise_img.noise(noise_type='gaussian', attenuate=noise)
        
            output_file = output_prefix + '_noise' + str(noise) +'_blur' + str(blur) + '.png'

            # Output the blurred/noisy image
            output_filename=os.path.join(augmented_output,output_file)
            blur_img.save(filename=output_filename)

            analysis = image_analysis(output_filename)

    
# Analysis function
def image_analysis(filename):
    print("Performing analysis of " + filename)
    
    # IQ Analysis
    iq_analysis(filename)

    # Facial
    facial_recognition(filename)

    # QR code
    qrcode_recognition(filename)

    # Text 
    text_recognition(filename)


    


    



# Facial recognition using OpenCV HAARCascade
# see https://medium.com/geeky-bawa/face-detection-using-haar-cascade-classifier-in-python-using-opencv-97873fbf24ec for details


facial_reference = 'faces_in_croud.jpg'
facial_cascPath = "haarcascade_frontalface_default.xml"
facial_debug = 0

def facial_recognition(faces_image):
    # read the image
    img = cv2.imread(faces_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(facial_cascPath)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    if facial_debug:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Faces found", img)
        cv2.waitKey(0)

    return len(faces)


# Text recoggnition using PYTesseract

text_reference = 'eye_chart.jpg'
text_correct = 'ZSHCHSKRNCHKRVDHONSDCVOKHDNRCSBDCLKZVHSROAHKGBCANOMPVESRPKUEOBTVXRMJHCAZDI'

def text_recognition(text_image):
    #text_image = os.path.join(reference_source, text_reference)
    # Custom Text Recognition config
    #custom_config = r'--oem 3 --psm 6'
    img=cv2.imread(text_image)
    #cv2.imshow("text found", img)
    #cv2.waitKey(0)
    output = pytesseract.image_to_string(img) #, config=custom_config)
    # remove linefeeds
    output = output.replace('\n','')
    if output == '':
        print("No text found")
        return 0
    elif output == text_correct:
        print("Text correctly identified")
        return len(text_correct)
    else:
        length_delta = len(output) - len(text_correct)
        print("Text misidentified,  " + str(length_delta) + " character delta")
        return len(output)
    #return [data, bbox, straight_qrcode]

    return output

# QR Code Recognition with OpenCV

qrcode_reference = 'qrcode.jpg'
qrcode_url = 'https://www.imatest.com/'

def qrcode_recognition(qrcode_image):
    #qrcode_image = os.path.join(reference_source, qrcode_reference)
    img = cv2.imread(qrcode_image)
    detector = cv2.QRCodeDetector()

    # detect and decode
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    if data == '':
        print("QR code not found!")
    elif data == qrcode_url:
        print("QR code correctly identified")
    else:
        print("QR code misidentified as " + data)
    return [data, bbox, straight_qrcode]


ini_file="imatest-v2.ini"
mean_part_way_idx =2
noise_metric = "SNR_BW_dB_RGBY"
R_channel = 0
G_channel = 1
B_channel = 2

def iq_analysis(filename):
    result = imatestLib.esfriso(input_file=filename,
                            root_dir=root,
                            op_mode=ImatestLibrary.OP_MODE_SEPARATE,
                            ini_file=ini_file)
    data = json.loads(result)
    mtf50_CP = data['esfrisoResults']['mtf50_CP_summary'][mean_part_way_idx]
    R_snr = data['esfrisoResults'][noise_metric][R_channel]
    G_snr = data['esfrisoResults'][noise_metric][G_channel]
    B_snr = data['esfrisoResults'][noise_metric][B_channel]
    SNR_mean = (R_snr + G_snr + B_snr) /3
    print("IQ Data:  MTF50 " + str(mtf50_CP) + " C/P, SNR: " + str(SNR_mean))
    

main()