import re
import os
import png
import glob
import pydicom
import pandas as pd
import pickle
import numpy as np
from multiprocessing import Pool
from pydicom.pixel_data_handlers.util import apply_voi_lut

def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth_output=12):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth output: what is the bitdepth of the output image you want!
    """
    try:
        ds = pydicom.read_file(dicom_filename)
        image = ds.pixel_array
        image = apply_voi_lut(image, ds, index = 0)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            image = 2**ds.BitsStored - 1 - image
        if bitdepth_output == 16:
            image = np.uint16(image)
        with open(png_filename, 'wb') as f:
            writer = png.Writer(
                height=image.shape[0],
                width=image.shape[1],
                bitdepth=ds.BitsStored,
                greyscale=True
            )
            writer.write(f, image.tolist())
    except Exception as e:
        print(e)
        print(dicom_filename)

def dicom_list_func(dicom_folder):
    #for dicom_folder in dicom_list:
    #print(count,'/', len(dicom_list))
    dicom_filenames = glob.glob(path_to_dicom+'/'+dicom_folder[1]+'/*.dicom')
    print("Id:{}, dicom_filenames:{}".format(dicom_folder[0], dicom_filenames))
    for dicom_filename in dicom_filenames:
        case_path = '/groups/dso/spathak/vindr/original_png_16bit/'+dicom_folder[1]
        if not os.path.exists(case_path):
            os.mkdir(case_path)
        png_filename = case_path+'/'+dicom_filename.split('/')[-1].strip('.dicom')+'.png'
        if not os.path.exists(png_filename):
            save_dicom_image_as_png(dicom_filename, png_filename, 16)

if __name__ == '__main__':
    path_to_dicom = "/groups/dso/spathak/vindr/physionet.org/files/vindr-mammo/1.0.0/images"
    dicom_list = os.listdir(path_to_dicom)
    dicom_list.remove('index.html')
    dicom_list1 = []
    for idx, x, in enumerate(dicom_list):
        dicom_list1.append([idx, x])
    p = Pool(10)
    p.map(dicom_list_func, dicom_list1)
    
    #path_to_monochrome2 = [0,'d9fad6c9f63a51b7555ad1f2600ee618']
    #dicom_list_func(path_to_monochrome2)
    #path_to_monochrome1 = [0, "f2b2d7d20e97526e1ce2e9b674d19640"]
    #dicom_list_func(path_to_monochrome1)

    #save_dicom_image_as_png(path_to_dicom+'/'+'1d6708c18a5e13bd1946bddd58b995f2/e186e7abebe74d4a0158303a98dd69cf.dicom', '/groups/dso/spathak/vindr/original_png_16bit/1d6708c18a5e13bd1946bddd58b995f2/e186e7abebe74d4a0158303a98dd69cf.png', 16)