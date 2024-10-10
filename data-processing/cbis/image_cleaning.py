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

def load_dicom(filename):
    "' Load a dicom file. If it is compressed, it is unzipped first. '"
    #print("filename:",filename)
    if (filename.endswith('.dcm')):
        ds = dicom.dcmread(filename)
    else:
        with gzip.open(filename) as fd:
            ds = dicom.dcmread(fd, force=True)
    #print(ds)
    #ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    return ds

def image_preprocessing(start_file, end_file, selected_image, input_path, modality, path_to_input_csvfile, output_folder_path):
    #============= Parameters =====================
    border_size = 105 #Border size
    blur = 21
    mask_dilate_iter = 20
    mask_erode_iter = 20
    sigma=0.33
    c_study=0
    df_img = {}
    
    #============= input ==================
    df=pd.read_csv(path_to_input_csvfile,sep=';')
    #print(df.columns)
    #df.loc[(~df['StudyInstanceUID'].isnull()) & (df['StoragePath']=='/mnt/data2/spathak/mammo_project_batch_2_png_24094-37553/MG'),['FullPath']]=df.loc[~df['StudyInstanceUID'].isnull()][['StoragePath', 'StudyInstanceUID']].agg('/'.join, axis=1)
    #df.loc[(~df['StudyInstanceUID'].isnull()) & (df['StoragePath']=='/mnt/data2/spathak/mammo_project_batch_2_png_24094-37553/MG'),['FullPath']]=df.loc[~df['StudyInstanceUID'].isnull()][['FullPath', 'AccessionNum']].astype('str').agg('_'.join, axis=1)
    study_total=df[~df['ImageName'].isnull()].shape[0]
    index_list=df.loc[~df['ImageName'].isnull()].index
    print(study_total)
    if selected_image!='':
        start_file = df[df['ImageName'].str.strip('_1.1')==selected_image].index[0]
        end_file = df[df['ImageName'].str.strip('_1.1')==selected_image].index[0] + 1
        print(start_file)
    else:
        if end_file==0:
            end_file=study_total
    
    #======= preprocess image ==========
    #try:
    for i in range(start_file,end_file):
        print("image number:{}/{}".format(i,study_total))
        row=index_list[i]
        #print(df.loc[row,'ImageName'])
        #study_acc_num_path=input_path+df.loc[row,'ImageName'].strip('_1.1')+'_1-1.png'
        image=df.loc[row,'ImageName'].strip('_1.1')+'_1-1.png'
        img_path=input_path+image
        print("image_path:",img_path)
        breast_side = df.loc[row,'Views'][0]
        '''series_list=os.listdir(study_acc_num_path)
        series_total=len(series_list)
        c_series=1
        print("Study:{}/{}".format(c_study,study_total))
        image_list=glob.glob(study_acc_num_path+'/**/*.png', recursive=True)
        print(image_list)
        image_total = len(image_list)
        '''
        c_image=1
        
        #for image in image_list:
        #    if '_processed.png' not in image and '.png' in image:
        #        print("Image:{}/{}".format(c_image,image_total))
        #img_path=image
        #-- Read image -----------------------------------------------------------------------
        img = cv2.imread(img_path, -1)
        print("original image:",img.shape)
        print("original image dtype:",img.dtype)
        try:
            height, width = img.shape
            #print(height, width)
        except:
            out1=open('./images_not_processed_ori_image_empty_MG.txt','a')
            out1.write(img_path+'/'+image+'\n')
            out1.close()
            continue
        img_copy=img.copy()
        #gray = (img_copy/256).astype('uint8')
        #gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        print("img8 shape:",gray.shape)
        print("img8 dtype:", gray.dtype)

        #== Processing =======================================================================
        if modality=='MG':
            #masking method
            filename = output_folder_path+'/'+image
            print("output_path:",filename)
            #try:
            png_name=filename#+'.png'
            #if os.path.exists(png_name):
            #    os.remove(png_name)
            processed_img8,img_mask,x,y,w,h = mask_image(c_image,gray,gray,sigma, mask_dilate_iter, mask_erode_iter,blur,border_size,breast_side)
            df_img[image.split('.')[0]] = [image.split('.')[0], x, y, x+w, y+h, img.shape[0], img.shape[1], processed_img8.shape[0], processed_img8.shape[1]]
            print("img8 shape:",processed_img8.shape)
            print("img8 dtype:",processed_img8.dtype)
            '''with open(png_name, 'wb') as f:
                writer = png.Writer(
                    height=processed_img8.shape[0],
                    width=processed_img8.shape[1],
                    bitdepth=8,
                    greyscale=True
                )
                writer.write(f, processed_img8.tolist())
            '''
            processed_img16 = image_16bit_preprocessing(img,img_mask,x,y,w,h,breast_side)
            print("img16 shape:",processed_img16.shape)
            print("img16 dtype:",processed_img16.dtype)
            '''with open('./'+selected_image+'.png', 'wb') as f:
                writer = png.Writer(
                    height=processed_img16.shape[0],
                    width=processed_img16.shape[1],
                    bitdepth=16,
                    greyscale=True
                )
                writer.write(f, processed_img16.tolist())
            '''
            '''cv2.namedWindow("Display frame"+str(c_image), cv2.WINDOW_NORMAL)
            cv2.imshow("Display frame"+str(c_image), processed_img)
            cv2.waitKey(0)'''
            #cv2.imwrite(png_name, processed_img)
            #df.loc[row,'png_processed']='Yes'
            '''except Exception as e1:
                df.loc[row,'png_processed']='No'
                out1=open('./error_in_processed_image_dicom_png_MG.txt','a')
                out1.write(img_path+'/'+image+','+'png'+','+str(e1)+'\n')
                out1.close()
            '''
            '''try:
                dicom_path=raw_dicom_path+'/'+df.loc[row]['StudyInstanceUID']+'/'+series.split('_')[1]+'/'+image.strip('.png')+'.dcm'
                npy_name=filename+'.npy'
                if os.path.exists(npy_name):
                    os.remove(npy_name)
                processed_dicom=dicom_preprocessing(img_mask,x,y,w,h,dicom_path,breast_side)
                np.save(npy_name,processed_dicom)
                df.loc[row,'dicom_processed']='Yes'
            except Exception as e2:
                df.loc[row,'dicom_processed']='No'
                out1=open('./error_in_processed_image_dicom_png_MG.txt','a')
                out1.write(series_path+'/'+image+','+'dicom'+','+str(e2)+'\n')
                out1.close()
            '''
        #c_image=c_image+1
        c_study+=1
    #df=df.drop(['FullPath'],axis=1)
    #df.to_csv(path_to_output_csvfile,sep=';',na_rep='NULL',index=False)
    df_img_pd = pd.DataFrame.from_dict(df_img, orient='index', columns=['ImageName', 'pro_min_x', 'pro_min_y', 'pro_max_x', 'pro_max_y', 'ori_height', 'ori_width', 'processed_height', 'processed_width'])
    #df_img_pd.to_csv(path_to_img_size,sep=';',na_rep='NULL',index=False)
    '''except Exception as e3:
        print("Exception",str(e3))
        print(i,row)
        #df=df.drop(['FullPath'],axis=1)
        df.to_csv(path_to_output_csvfile,sep=';',na_rep='NULL',index=False)
    '''        

def mask_image(disp_id,gray,img,sigma,MASK_DILATE_ITER,MASK_ERODE_ITER,BLUR,border_size,breast_side):
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
    edges = cv2.Canny(gray, 0, 10)

    #edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    edges = edges[border_size:-border_size,border_size:-border_size]
    edges = cv2.GaussianBlur(edges, (BLUR, BLUR), 0)
    '''cv2.imshow("Display frame"+str(disp_id),edges)
    cv2.waitKey(0)
    '''
    #for cbis
    edges = cv2.copyMakeBorder(edges,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,value=0)
    
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

    cimg = cv2.dilate(cimg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), None, iterations=MASK_DILATE_ITER)
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
    elif breast_side=='R':
        crop_img = res[y:y+h, max(0,x-20):x+w]
    else:
        crop_img = res[y:y+h, x:x+w]
    #print(gray.shape)
    print("crop img:",crop_img.shape)
    '''cv2.imshow("Display frame"+str(disp_id),crop_img)
    cv2.waitKey(0)
    '''
    
    return crop_img,cimg,x,y,w,h

def image_16bit_preprocessing(img16,img_mask,x,y,w,h,breast_side):
    res=cv2.bitwise_and(img16, img16, mask = img_mask)
    if breast_side=='L':
        res_crop = res[y:y+h, x:x+w+20]
    elif breast_side=='R':
        res_crop = res[y:y+h, x-20:x+w]
    else:
        res_crop = res[y:y+h, x:x+w]
    '''print(res_crop[:200,:200])
    plt.imshow(res_crop,cmap=plt.cm.gray)
    plt.show()
    '''
    return res_crop

def dicom_preprocessing(img_mask,x,y,w,h,dicom_path,breast_side):
    if modality=='MG':
        #print(dicom_path)
        ds=load_dicom(dicom_path)
        ds_pixel_array=ds.pixel_array
        #print('pixel array shape:',ds_pixel_array.shape)
        #plt.imshow(ds_pixel_array, cmap=plt.cm.gray)
        #plt.show()
        #tensor_dicom_scaled = ds.scaled_px
        '''print(ds.RescaleIntercept)
        print(ds.RescaleSlope)
        print(ds.Rows)
        print(ds.Columns)
        print(ds.PhotometricInterpretation)
        print(tensor_dicom_scaled)'''
        #tensor_dicom_scaled=tensor_dicom_scaled.unsqueeze(0)
        #print('pixel array scaled:',tensor_dicom_scaled.shape)
        #plt.imshow(tensor_dicom_scaled,cmap=plt.cm.gray)
        #plt.show()
        res=cv2.bitwise_and(ds_pixel_array, ds_pixel_array, mask = img_mask)
        '''plt.imshow(res,cmap=plt.cm.gray)
        plt.show()'''
        #print('processed dicom shape',res.shape)
        if breast_side=='L':
            res_crop = res[y:y+h, x:x+w+20]
        elif breast_side=='R':
            res_crop = res[y:y+h, x-20:x+w]
        else:
            res_crop = res[y:y+h, x:x+w]
        #plt.imshow(res_crop,cmap=plt.cm.gray)
        #plt.show()
        #plt.hist(res.flatten(), color='c')
        #plt.show()
        #input('wait')
        return res_crop

def read_imgsize_csvfile(path_to_img_size):
    df = pd.read_csv(path_to_img_size, sep=';')
    df['BreastSide'] = df['ImageName'].str.split('_').str[3].map({'LEFT':'L', 'RIGHT':'R'})
    df['pro_max_x'] = df.apply(lambda x: x['pro_max_x'] + 20 if x['BreastSide']=='L' else x['pro_max_x'], axis=1)
    df['pro_min_x'] = df.apply(lambda x: x['pro_min_x'] - 20 if x['BreastSide']=='R' else x['pro_min_x'], axis=1)
    df.to_csv(path_to_img_size1, sep=';', na_rep='NULL', index=False)

#Initialization
modality='MG'

input_path='/projects/dso_mammovit/project_kushal/data/raw_images_16bit/'
#input_path='D:/PhD-UT-laptop-backup/PhD/projects/radiology breast cancer/cbis-ddsm/raw-images/'

#this is a csv file with the following structure: AccessionNum;StudyDate;Groundtruth;Modality;StudyInstanceUID;StoragePath;dicom_processed;png_processed;Views
#This is a file containing each study, the location of the images belonging to that study, the groyundtruth, the number of views in that study. This file will be passed as input to the image_preprocessing function
path_to_input_csvfile = '/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_singleinstance_groundtruth.csv'
path_to_output_csvfile = '/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_singleinstance_groundtruth_1.csv'
#path_to_input_csvfile = 'C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/cbis-ddsm/MG_training_files_cbis-ddsm_singleinstance_groundtruth.csv'
#path_to_output_csvfile = 'C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/cbis-ddsm/MG_training_files_cbis-ddsm_singleinstance_groundtruth_1.csv'
path_to_img_size = '/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_imgpreprocessing_size_correctscaling8bit.csv'
path_to_img_size1 = '/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv'

# path for output folder - png files
output_folder_path = '/projects/dso_mammovit/project_kushal/data/processed_owncleaningalgo_8bit_correctscaling'
#output_folder_path = 'C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/cbis-ddsm/processed-images-gmic'

start = 0
end = 0
#selected_image='Mass-Training_P_00001_LEFT_MLO'#Calc-Training_P_01458_LEFT_CC'

#image_preprocessing based on modality
image_preprocessing(start, end, selected_image, input_path, modality, path_to_input_csvfile, output_folder_path)

#read_imgsize_csvfile(path_to_img_size)