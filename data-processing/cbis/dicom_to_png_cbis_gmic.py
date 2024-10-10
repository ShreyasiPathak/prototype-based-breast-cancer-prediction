import png
import pydicom
import os
import glob
import re
import pandas as pd
import pickle

def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
                     Set to 16 for 16-bit mammograms, etc.
                     Make sure you are using correct bitdepth!
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    print(pydicom.dcmread(dicom_filename))
    input('halt')
    with open(png_filename, 'wb') as f:
        writer = png.Writer(
            height=image.shape[0],
            width=image.shape[1],
            bitdepth=bitdepth,
            greyscale=True
        )
        writer.write(f, image.tolist())

path_to_dicom = "/projects/dso_mammovit/project_kushal/data/raw-dicom/CBIS-DDSM/"
dicom_list = os.listdir(path_to_dicom)
dicom_list = list(filter(lambda x: not re.search('_\d$',x), dicom_list))

#path_to_csv='/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_roi_groundtruth.csv'
#df=pd.read_csv(path_to_csv, sep=';')

#taking a sample from the GMIC format pickle input file
'''dicom_list=[]
f = open('/homes/spathak/GMIC/sample_data/exam_list_before_cropping.pkl','rb')
pkl_file = pickle.load(f)
for dic_item in pkl_file:
    dic_keys = dic_item.keys()
    print(dic_item)
    if 'L-CC' in dic_keys:
        dicom_list.append('_'.join(dic_item['L-CC'][0].split('_')[:-1]))
    if 'L-MLO' in dic_keys:
        dicom_list.append('_'.join(dic_item['L-MLO'][0].split('_')[:-1]))
    if 'R-CC' in dic_keys:
        dicom_list.append('_'.join(dic_item['R-CC'][0].split('_')[:-1]))
    if 'R-MLO' in dic_keys:
        dicom_list.append('_'.join(dic_item['R-MLO'][0].split('_')[:-1]))
print(dicom_list)
input('halt')
'''
for dicom_folder in dicom_list:
    dicom_filename = glob.glob(path_to_dicom+'/'+dicom_folder+'/**/*.dcm', recursive=True)[0]
    print(dicom_filename)
    png_filename = '/projects/dso_mammovit/project_kushal/data/raw-images-gmic/'+dicom_folder+'_'+dicom_filename.split('/')[-1].strip('.dcm')+'.png'
    print(png_filename)
    save_dicom_image_as_png(dicom_filename,png_filename,16)
    #input('halt')