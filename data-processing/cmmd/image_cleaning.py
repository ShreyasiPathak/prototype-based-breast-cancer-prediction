# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:46:07 2022

@author: PathakS
"""

import os
import re
import cv2
import glob
import png
import numpy as np
import pandas as pd
import pydicom as dicom
import gzip
import matplotlib.cm as cm
from matplotlib import pyplot as plt

def visualize_img(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def image_preprocessing(input_path, path_to_input_csvfile, output_folder_path_8bit, output_folder_path_16bit):
    #======= preprocess image ==========
    case_list = os.listdir(input_path)
    study_total = len(case_list)
    df = pd.read_csv(path_to_input_csvfile, sep = ';')
    df_img = {}
    for i, case_folder in enumerate(case_list):
        print("case number:{}/{}".format(i, study_total))
        case_path = input_path + '/' + case_folder
        image_list = os.listdir(case_path)
        c_image = 1
        if not os.path.exists(output_folder_path_8bit + '/' + case_folder):
            os.mkdir(output_folder_path_8bit + '/' + case_folder)
        #if not os.path.exists(output_folder_path_16bit + '/' + case_folder):
        #    os.mkdir(output_folder_path_16bit + '/' + case_folder)
        if case_folder == 'D1-1343':
            for image in image_list:
                print("image number:{}/{}".format(c_image, len(image_list)))
                img_path = case_path + '/' + image

                print("image_path:",img_path)

                if df[df['OriImgPath'].str.contains(img_path)].shape[0] == 1:
                    view = df.loc[df['OriImgPath'].str.contains(img_path), 'Views'].values[0]
                    breast_side = view[0]
                    img = cv2.imread(img_path, -1)
                    print("original image:",img.shape)
                    print("original image dtype:",img.dtype)
                
                try:
                    height, width = img.shape
                    #print(height, width)
                    img_copy = img.copy()
                    gray = (img_copy/256).astype('uint8')
                    #gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U) #image conversion to 8bit
                    #visualize_img(gray[1759:, 1400:])

                    #masking method
                    filename_8bit = output_folder_path_8bit + '/' + case_folder + '/' + view + '_' + image
                    df.loc[df['OriImgPath'].str.contains(img_path), 'ShortPath'] = case_folder + '/' + view + '_' + image
                    print("output_path 8 bit:", filename_8bit)
                    #if not os.path.exists(filename_8bit):
                    processed_img8, img_mask, min_x, min_y, max_x, max_y = mask_image(c_image, gray, gray, breast_side)
                    df_img[case_folder + '/' + view + '_' + image] = [case_folder + '/' + view + '_' + image, min_x, min_y, max_x, max_y, img.shape[0], img.shape[1], processed_img8.shape[0], processed_img8.shape[1]]
                    
                    #print("img8 shape:", processed_img8.shape)
                    #print('img8 dtype', processed_img8.dtype)
                    with open(filename_8bit, 'wb') as f:
                        writer = png.Writer(
                            height=processed_img8.shape[0],
                            width=processed_img8.shape[1],
                            bitdepth=8,
                            greyscale=True
                        )
                        writer.write(f, processed_img8.tolist())
                    
                    '''filename_16bit = output_folder_path_16bit + '/' + case_folder + '/' + view + '_' + image
                    print("output_path 16 bit:", filename_16bit)
                    if not os.path.exists(filename_16bit):
                        processed_img16 = image_16bit_preprocessing(img, img_mask, x, y, w, h, breast_side)
                        #print("img16 shape:", processed_img16.shape)
                        #print("img16 dtype:", processed_img16.dtype)
                        with open(filename_16bit, 'wb') as f:
                            writer = png.Writer(
                                height=processed_img16.shape[0],
                                width=processed_img16.shape[1],
                                bitdepth=16,
                                greyscale=True
                            )
                            writer.write(f, processed_img16.tolist())
                    df.loc[df['image_id'].str.contains(image.split('.')[0]), 'ShortPath'] = case_folder + '/' + view + '_' + image
                    '''
                    #input('halt')
                
                except Exception as e:
                    print(e)
                    out1=open('./images_not_processed_ori_image_empty_MG.txt','a')
                    out1.write(img_path+'/'+image+'\n')
                    out1.close()
                
                #cv2.imwrite(png_name, processed_img)
                c_image=c_image+1
    df_img_pd = pd.DataFrame.from_dict(df_img, orient='index', columns=['ImageName', 'pro_min_x', 'pro_min_y', 'pro_max_x', 'pro_max_y', 'ori_height', 'ori_width', 'processed_height', 'processed_width'])
    df_img_pd.to_csv(path_to_img_size,sep=';',na_rep='NULL',index=False)
    df.to_csv(path_to_output_csvfile, sep=';', na_rep='NULL', index=False)

def mask_image(disp_id, gray, img, breast_side):
    #============= Parameters ===========
    border_size = 105 #Border size
    BLUR = 21
    MASK_DILATE_ITER = 20
    MASK_ERODE_ITER = 20
    sigma=0.33

    '''cv2.namedWindow("Display frame"+str(disp_id), cv2.WINDOW_NORMAL)
    cv2.imshow("Display frame"+str(disp_id),gray)
    cv2.waitKey(0)
    '''
    #-- Edge detection -------------------------------------------------------------------
    v = np.median(gray)
    
    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    #for zgt
    #edges = cv2.Canny(gray, lower, upper)
    
    #for cbis
    edges = cv2.Canny(gray, 0, 5)

    #edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    edges = cv2.GaussianBlur(edges, (BLUR, BLUR), 0)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    #for cbis
    #edges = cv2.copyMakeBorder(edges, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=0)
    
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), None, iterations=MASK_ERODE_ITER)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), None, iterations=MASK_DILATE_ITER)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), None, iterations=MASK_DILATE_ITER)
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), None, iterations=MASK_ERODE_ITER)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    
    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for idx,c in enumerate(contours):
        contour_info.append((
            idx,
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[3], reverse=True)
    max_contour = contour_info[0][1]
    
    cimg = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(cimg, [max_contour], -1, color=(255,255,255), thickness=-1)
    '''cv2.imshow("Display frame"+str(disp_id),cimg)
    cv2.waitKey(0)
    '''

    #cimg = cv2.dilate(cimg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), None, iterations=MASK_DILATE_ITER)
    '''cv2.imshow("Display frame"+str(disp_id),cimg)
    cv2.waitKey(0)
    '''
    '''print(cimg.shape)
    print(img.shape)'''
    res = cv2.bitwise_and(img, img, mask = cimg)
    '''print(res.shape)
    cv2.imshow("Display frame"+str(disp_id),res)
    cv2.waitKey(0)
    '''
    #print(res.shape)
    #x,y,w,h=cv2.boundingRect(res[:,:,0])
    #print(res)
    #print(res.dtype)
    x,y,w,h=cv2.boundingRect(res[:,:])
    #print(x,y)
    #print(res.shape)
    #print(x,y,w,h)
    if breast_side=='L':
        crop_img = res[y:y+h, x:min(x+w+20,res.shape[1])]
        min_y = y
        max_y = y+h
        min_x = x
        max_x = min(x+w+20,res.shape[1])
    elif breast_side=='R':
        crop_img = res[y:y+h, max(0,x-20):x+w]
        min_y = y
        max_y = y+h
        min_x = max(0,x-20)
        max_x = x+w
    else:
        crop_img = res[y:y+h, x:x+w]
        min_y = y
        max_y = y+h
        min_x = x
        max_x = x+w
    #print(gray.shape)
    print("crop img:",crop_img.shape)
    '''cv2.imshow("Display frame"+str(disp_id),crop_img)
    cv2.waitKey(0)
    '''
    
    return crop_img,cimg,min_x,min_y,max_x,max_y

def image_16bit_preprocessing(img16, img_mask, x, y, w, h, breast_side):
    res=cv2.bitwise_and(img16, img16, mask = img_mask)
    if breast_side=='L':
        res_crop = res[y:y+h, x:min(x+w+20,res.shape[1])]
    elif breast_side=='R':
        res_crop = res[y:y+h, max(0,x-20):x+w]
    else:
        res_crop = res[y:y+h, x:x+w]
    '''print(res_crop[:200,:200])
    plt.imshow(res_crop,cmap=plt.cm.gray)
    plt.show()
    '''
    return res_crop

def sort_file(path_to_input_csvfile):
    df = pd.read_csv(path_to_input_csvfile, sep=';')
    df = df.sort_values(by=['Patient_Id', 'Views'])
    df.to_csv(path_to_output_csvfile, sep=';', na_rep='NULL', index=False)


input_path='/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/original_png_8bit'
#input_path='D:/PhD-UT-laptop-backup/PhD/projects/radiology breast cancer/cbis-ddsm/raw-images/'

#this is a csv file with the following structure: AccessionNum;StudyDate;Groundtruth;Modality;StudyInstanceUID;StoragePath;dicom_processed;png_processed;Views
#This is a file containing each study, the location of the images belonging to that study, the groyundtruth, the number of views in that study. This file will be passed as input to the image_preprocessing function
path_to_input_csvfile = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/cmmd_singleinstance_groundtruth.csv'
path_to_output_csvfile = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/cmmd_singleinstance_groundtruth1.csv'

# path for output folder - png files
output_folder_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/processed_png_8bit'

path_to_img_size = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/cmmd_singleinstance_imgpreprocessing_size1.csv'

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

#if not os.path.exists(output_folder_path_16bit):
#    os.mkdir(output_folder_path_16bit)

#image_preprocessing based on modality
image_preprocessing(input_path, path_to_input_csvfile, output_folder_path, None)

sort_file(path_to_input_csvfile)