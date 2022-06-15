# Correlating the performance of computer vision algorithms relative to objective image quality
# This example creates a set of augmented images that can be analyzed for how well they can be used for image 
import numpy as np
from wand.image import Image
from pgmagick.api import PmagickImage
import cv2
import sys
import os
import pytesseract



root = '.'
reference_source = 'Reference-Images'
augmented_output = 'Augmented-Images'
output_prefix = 'output_image'
def prepare_montage():
    images = [facial_reference, ]
    return



esfriso_reference = 'esfriso.png'

# Image simulation using wand

def image_simulation():
    imaage_file = os.path.join(root,reference_source,esfriso_reference)
    with Image(imaage_file) as img:

        # Iterate through different levels of blur
        for blur in range(0,5,1):
            blur_img = img
            blur_img.blur(blur,blur+1)

        
            # Iterate through different levels of noise
            for noise in np.arange(0.0,1.0,0.2):
                noise = np.around(noise, decimals=1)
                noise_img=blur_img
                noise_img.noise(noise_type='gaussian', attenuate=noise)

                output_file = output_prefix + '_noise' + str(noise) +'_blur' + str(blur) + '.png'

                # Output the blurred/noisy image
                output_filename=os.path.join(augmented_output,output_file)
                noise_img.save(filename=output_filename)

                analysis = image_analysis(output_filename)

    

def image_analysis(filename):
    print("Performing analysis of " + filename)
    


# Facial recognition using OpenCV HAARCascade

facial_reference = 'faces_in_croud.jpg'
facial_cascPath = "haarcascade_frontalface_default.xml"

def facial_recognition():
    faces_image = os.path.join(reference_source, facial_reference)
    
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

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", img)
    cv2.waitKey(0)
    return faces


# Text recoggnition using PYTesseract

text_reference = 'eye_chart.jpg'

def text_recognition():
    text_image = os.path.join(reference_source, text_reference)
    # Custom Text Recognition config
    #custom_config = r'--oem 3 --psm 6'
    img=cv2.imread(text_image)
    #cv2.imshow("text found", img)
    #cv2.waitKey(0)
    output = pytesseract.image_to_string(img) #, config=custom_config)
    return output


qrcode_reference = 'qrcode.jpg'

def qrcode_recognition():
    qrcode_image = os.path.join(reference_source, qrcode_reference)
    img = cv2.imread(qrcode_image)
    detector = cv2.QRCodeDetector()

    # detect and decode
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    return [data, bbox, straight_qrcode]


def main():
    #image_simulation()
    #facial_recognition()
    #print(text_recognition())
    data, bbox, straight_qrcode = qrcode_recognition()
    print(data)



main()