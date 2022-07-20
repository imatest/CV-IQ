# Correlating the performance of computer vision algorithms relative to objective image quality
# This example creates a set of augmented images that can be analyzed for how well they can be used for image 
import numpy as np                     # https://numpy.org/
from wand.image import Image           # Wand imagemagic python https://docs.wand-py.org/en/0.6.7/
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2                             # OpenCV-python https://pypi.org/project/opencv-python/
import pytesseract                     # OCR engine https://pypi.org/project/pytesseract/
import difflib
from imatest.it import ImatestLibrary, ImatestException  # Imatest IT https://www.imatest.com/docs/imatest-it-instructions/#Python for instructions on Imatest IT
import sys
import subprocess                      # for calling ImageMagick
import os
import json
from mpl_toolkits.mplot3d import Axes3D

root = '.'
reference_source = 'Reference-Images'
augmented_output = 'Augmented-Images'
output_prefix = 'output_image'
plot_output = "Plot-Output"
imatestLib = []

# Main Function
def main():
    global imatestLib
    if imatest_analysis_enabled:
        imatestLib = ImatestLibrary()

    #prepare_montage()
    image_simulation()
    if imatest_analysis_enabled:
        imatestLib.terminate_library()



# Montage of images containing all target types, using imagemagick's montage command line tool
# Install imagemagick here: https://imagemagick.org/index.php
# would have used wand but this issue exists: https://github.com/emcconville/wand/issues/575

montage_reference = 'montage.png'
def prepare_montage():
    images = [facial_reference, imatest_reference, text_reference, qrcode_reference]
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


# Image simulation using wand-python, with image quality analysis by Imatest
# See https://wand-py.org/ https://imatest.com/

imatest_enabled=1
imatest_analysis_enabled=0
facial_enabled=0
qrcode_enabled=0
text_enabled=0

def image_simulation():
    
    montage_image_file = os.path.join(root,augmented_output,montage_reference)
    simulations=[]
    if imatest_enabled:
        if imatest_analysis_enabled:
            simulations.append({'type':'imatest', 'image_file':imatest_reference, 'analysis':iq_analysis})
        else:
            simulations.append({'type':'imatest', 'image_file':imatest_reference})
    if facial_enabled:
        simulations.append(    {'type':'facial',  'image_file':facial_reference,  'analysis':facial_recognition})
    if qrcode_enabled:
        simulations.append(    {'type':'qrcode',  'image_file':qrcode_reference,  'analysis':qrcode_recognition})
    if text_enabled:
        simulations.append(    {'type':'text',    'image_file':text_reference,    'analysis':text_recognition})
 
    output = {}
    arbitrary_noise_levels = np.arange(0.0,1.1,0.1)
    arbitrary_blur_levels = np.arange(0.0,5.0,0.5)
    objective_noise_levels = []
    objective_blur_levels = []

    # create folders for output
    for simulation in simulations:
        output_folder = os.path.join(root,augmented_output,simulation['type'])
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output[simulation['type']] = {}

        if simulation['type'] == 'facial':
            face_data = []
        if simulation['type'] == 'qrcode':
            qrcode_data = []
        if simulation['type'] == 'text':
            text_data = []
        elif simulation['type'] == 'imatest':
            snr_data = []
            mtf_data = []

        # Iterate through different levels of blur
        blur_index = 1
        for blur in arbitrary_blur_levels:
            output[simulation['type']][blur] = {}
            simulation_image = Image(filename=os.path.join(root, reference_source, simulation['image_file']))
            simulation_image.blur(radius=0,sigma=blur)

            if simulation['type'] == 'facial':
                face_row = []
            elif simulation['type'] == 'qrcode':
                qrcode_row = []
            elif simulation['type'] == 'text':
                text_row = []
            elif simulation['type'] == 'imatest':
                snr_row = []
                mtf_row = []

            # Iterate through different levels of noise
            noise_index = 1
            for noise in arbitrary_noise_levels:
                output[simulation['type']][blur][noise] = {}
                noise = np.around(noise, decimals=1)
                noise_img=simulation_image
                noise_img.noise(noise_type='gaussian', attenuate=noise)
            
                output_file = simulation['type'] + '_noise' + str(noise) +'_blur' + str(blur) + '.png'

                # Output the blurred/noisy image
                output_filename=os.path.join(output_folder,output_file)
                noise_img.save(filename=output_filename)

                # 
                if "analysis" in simulation:
                    analysis_output = (simulation['analysis'])(output_filename)

                    output[simulation['type']][blur][noise] = analysis_output

                    if simulation['type'] == 'facial':
                        face_row.append(analysis_output)
                    if simulation['type'] == 'qrcode':
                        qrcode_row.append(analysis_output)
                    if simulation['type'] == 'text':
                        text_row.append(analysis_output)
                    if simulation['type'] == 'imatest':
                        snr_row.append(analysis_output['snr'])
                        if analysis_output['mtf50'] == '_NaN_':         # some MTFs came as NAN and should be plotted as 0
                            sharpness_metric = 0
                            print('WARNING: NaN in sfr calc from ' + output_filename)
                        else:
                            sharpness_metric = analysis_output['mtf50']    # Choice of MTF50 is arbitrary, other sharpness metrics could correlate better with particular items
                        mtf_row.append(sharpness_metric)
                        if noise_index == 1:
                            objective_blur_levels.append(sharpness_metric)
                noise_index += 1
            

            if simulation['type'] == 'facial':
                face_data.append(face_row)
            if simulation['type'] == 'qrcode':
                qrcode_data.append(qrcode_row)
            if simulation['type'] == 'text':
                text_data.append(text_row)
            elif simulation['type'] == 'imatest':
                snr_data.append(snr_row)
                mtf_data.append(mtf_row)
                if blur_index == 1:             # use first row of unblured noise simulations for SNR axis
                    objective_noise_levels = snr_row
                
            blur_index += 1
                
        X_arbitrary, Y_arbitrary = np.meshgrid(arbitrary_noise_levels, arbitrary_blur_levels)
        X_label_arbitrary = "Noise level"
        Y_label_arbitrary = "Blur level"
        
        if imatest_enabled:         # key into SNR and MTF50 as key axes if we can calcualte it with Imatest
            X_objective, Y_objective = np.meshgrid(objective_noise_levels, objective_blur_levels)
            X = X_objective
            Y = Y_objective
            X_label = "Mean SNR"
            Y_label = "MTF50 Cycles/Pixel"
        else:                       # without objective image quality analysis we are made to depend on arbitrary values
            X = X_arbitrary
            Y = Y_arbitrary
            X_label = X_label_arbitrary
            Y_label = Y_label_arbitrary

        #surface_plot_data(X, Y, np.array(face_data), xlabel="")
        if simulation['type'] == 'facial':
            surface_plot_data(X, Y, np.array(face_data), title="Faces Found", xlabel=X_label, ylabel=Y_label, azimuth=-123)
        elif simulation['type'] == 'imatest':           # Use arbitrary units for ploting to see relation between arbitrary and objective
            if imatest_analysis_enabled:
                surface_plot_data(X_arbitrary, Y_arbitrary, np.array(snr_data), title="Imatest SNR", xlabel=X_label_arbitrary, ylabel=Y_label_arbitrary)
                surface_plot_data(X_arbitrary, Y_arbitrary, np.array(mtf_data), title="Imatest SFR MTF50 C/P", xlabel=X_label_arbitrary, ylabel=Y_label_arbitrary)
        elif simulation['type'] == 'qrcode':
            surface_plot_data(X, Y, np.array(qrcode_data), title="QR Code Recongnition Success", xlabel=X_label, ylabel=Y_label, azimuth=-123)
        elif simulation['type'] == 'text':
            surface_plot_data(X, Y, np.array(text_data), title="Text Identification Success", xlabel=X_label, ylabel=Y_label, azimuth=-123)
    
    plt.waitforbuttonpress()
            


def surface_plot_data(X, Y, Z, *, title="", xlabel="", ylabel="", azimuth=45, elevation=20):
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.jet,linewidth=0.1, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.view_init(azim=azimuth, elev=elevation)
    
    plt.savefig(os.path.join(root,plot_output,title.replace(" ","_").replace("/","P")+".png"))
    plt.show()

# Facial recognition using OpenCV HAARCascade
# see https://medium.com/geeky-bawa/face-detection-using-haar-cascade-classifier-in-python-using-opencv-97873fbf24ec for details

# Faical reference selected from a crop of Croud by Paul Sableman https://www.flickr.com/photos/pasa/17412732639
facial_reference = 'Paul_Sableman_Crowd.png'
facial_cascPath = os.path.join("haarcascades","haarcascade_frontalface_alt.xml")
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

########
# Text recoggnition using PYTesseract
#

text_reference = 'eye_chart.jpg'
text_extremely_correct = 'ZSHCHSKRNCHKRVDHONSDCVOKHDNRCSVHDNKUOSRCBDCLKZVHSROAHKGBCANOMPVESR'

def text_recognition(text_image):
    #text_image = os.path.join(reference_source, text_reference)
    # Custom Text Recognition config
    #custom_config = r'--oem 3 --psm 6'
    img=cv2.imread(text_image)
    text_recognized = pytesseract.image_to_string(img) #, config=custom_config)
    # remove linefeeds
    text_recognized = text_recognized.replace('\n','')
    if text_recognized == '':
        print("No text found")
        success= 0
    else:
        success = text_accuracy(text_recognized,text_extremely_correct)
        print("Accuracy " + str(success) + ": \"" + text_recognized + "\"")
    #return [data, bbox, straight_qrcode]
    return success

def text_accuracy(s1,s2):
    s = difflib.SequenceMatcher(None,s1,s2)
    matched = 0
    for block in s.get_matching_blocks():
        matched += block.size

    return matched


########
# QR Code Recognition with OpenCV
#
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
        output = 0.0
    elif data == qrcode_url:
        print("QR code correctly identified")
        output = 1.0
    else:
        print("QR code misidentified as " + data)
        output = 0.25
    
    return output
    #return [data, bbox, straight_qrcode]



########
# IQ Analysis with Imatest
#
imatest_reference = 'esfriso.png'
max_snr = 50   # to prevent noise-free images from scoring too high we put a cap on SNR

def iq_analysis(filename):
    global imatestLib
    ini_file=os.path.join(root, r"imatest-v2.ini")
    noise_metric = "SNR_BW_dB_RGBY"
    R_channel = 0
    G_channel = 1
    B_channel = 2

    # normally we would call esfriso instead, but since this must work with high amounts of degredation we are using sfr and stepchart that have manual region selections
    sfr_data = imatest_analysis(imatestLib.sfr_json, filename, ini_file)

    # we pull the mtf50 from the first slanted edge as our sharpness metric
    mtf50_CP = sfr_data['sfrResults']['mtf50'][0]

    stepchart_data = imatest_analysis(imatestLib.stepchart_json, filename, ini_file)
    # we pull mean SNR_BW from red,green,blue as our noise metric
    SNR_mean = (stepchart_data['stepchartResults'][noise_metric][R_channel] + 
                stepchart_data['stepchartResults'][noise_metric][G_channel] + 
                stepchart_data['stepchartResults'][noise_metric][B_channel] ) / 3
    SNR_mean = min(SNR_mean, max_snr)

    print("IQ Data:  MTF50 " + str(mtf50_CP) + " C/P, SNR: " + str(SNR_mean))
    return {'mtf50':mtf50_CP, 'snr':SNR_mean}
    
# This function calls an Imatest IT library function module (run) on a specified image with associated ini settings
def imatest_analysis(run, image, ini_file):
    try:
        result = (run)(input_file=image,
                                root_dir=root,
                                op_mode=ImatestLibrary.OP_MODE_SEPARATE,
                                ini_file=ini_file)
    except ImatestException as iex:
        if iex.error_id == ImatestException.FloatingLicenseException:
            print("All floating license seats are in use.  Exit Imatest on another computer and try again.")
        elif iex.error_id == ImatestException.LicenseException:
            print("License Exception: " + iex.message)
        else:
            print(iex.message)
        exit_code = iex.error_id
    except Exception as ex:
        print(str(ex))
        exit_code = 2
    data = json.loads(result)
    return data

# Run the main function
main()