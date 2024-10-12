import os
import re
import cv2
import glob
import png
import pydicom
import numpy as np
import pandas as pd
import pydicom as dicom
import gzip
import matplotlib.cm as cm
from matplotlib import pyplot as plt

def calculate_image_size(input_folder):
    w_all=[]
    h_all=[]
    #h_w_all = []
    count_less=0
    count_more=0
    count_wless=0
    count_wmore=0

    case_list = os.listdir(input_folder)
    
    for i, case in enumerate(case_list):
        print(i, '/', len(case_list))
        case_path = input_folder + '/' + case
        image_list = os.listdir(case_path)
        for image in image_list:
            image_path = case_path + '/' + image
            img = cv2.imread(image_path, -1)
            h, w = img.shape
            #h_w_all.append([h,w])
            w_all.append(w)
            h_all.append(h)
            if w<1920 and h<2944:
                count_less+=1
            elif w>1920 and h>2944:
                count_more+=1
            elif w<1920 and h>2944:
                count_wless+=1
            elif w>1920 and h<2944:
                count_wmore+=1
    
    print("min w:", min(w_all))
    print("min h:", min(h_all))
    print("max w:", max(w_all))
    print("max h:", max(h_all))
    print("less than 1600,1600:",count_less)
    print("more than 1600,1600:",count_more)
    print("w less than 1600, h more than 1600:",count_wless)
    print("w more than 1600, h less than 1600:",count_wmore)
    w_mean_dataset = np.mean(np.array(w_all))
    w_std_dataset = np.std(np.array(w_all))
    h_mean_dataset = np.mean(np.array(h_all))
    h_std_dataset = np.std(np.array(h_all))
    return w_mean_dataset, w_std_dataset, h_mean_dataset, h_std_dataset

#check how many images are missing and from where
def sanity_check(input_folder):
    case_list = os.listdir(input_folder)
    print("total cases:", len(case_list))

    for i, case in enumerate(case_list):
        print(i, '/', len(case_list))
        case_path = input_folder + '/' + case
        image_list = os.listdir(case_path)
        views_saved = sorted([image.split('_')[0] for image in image_list])
        #print(image_list)
        #input('halt1')
        #if len(image_list) != 4:
        #    print("case:", case)
        #    input('halt')
        if views_saved!=['LCC', 'LMLO', 'RCC', 'RMLO']:
            print(case)
            print(views_saved)
            input('halt')

def aggregating_patient_case_info(grp, dic_groundtruth):
    grp_agg = pd.DataFrame(columns=['StudyInstanceUID', 'Views', 'BreastDensity', 'BIRADS', 'Groundtruth', 'ShortPath', 'Split'], index=range(1))
    grp_agg['StudyInstanceUID'] = np.unique(grp['study_id'])
    #print(grp['breast_density'].str.split(' ').str[-1])
    grp_agg['BreastDensity'] = ','.join(list(np.array(grp['breast_density'].str.split(' ').str[-1])))
    grp_agg['Views'] = "+".join(sorted(list(grp['Views'])))
    grp_agg['BIRADS'] = ",".join(i.split(' ')[-1] for i in list(grp['breast_birads']))
    grp_agg['MaxBirads'] = max(list(grp['breast_birads']))
    grp['Groundtruth'] = [dic_groundtruth[int(i.split(' ')[-1])] for i in list(grp['breast_birads'])]
    grp_agg['Split'] = np.unique(grp['split'])
    if len(np.unique(grp['Groundtruth']))>1:
        if 'malignant' in list(np.unique(grp['Groundtruth'])):
            grp_agg['Groundtruth']='malignant'
    else:
        grp_agg['Groundtruth']=np.unique(grp['Groundtruth'])
    grp_agg['ShortPath'] = np.unique(grp['study_id'])
    return grp_agg

#creating mammogram csv file
def create_MIL_csv_file(path_to_input_csvfile):
    dic_groundtruth = {1:'benign', 2:'benign', 3:'benign', 4:'malignant', 5:'malignant'}
    df = pd.read_csv(path_to_input_csvfile, sep = ';')
    df['Views'] = df['laterality'] + df['view_position']
    df_case = df.groupby(by=['study_id'], as_index=False, group_keys=False).apply(aggregating_patient_case_info, dic_groundtruth)
    print(df_case.groupby(by='Groundtruth').size())
    # open the file in the write mode
    '''for idx in df.index:
        print(idx,'/', len(df.index))
        if df.loc[idx, 'study_id'] not in dic_case.keys():
            dic_case[df.loc[idx, 'study_id']] = []
        dic_case[df.loc[idx, 'study_id']].append(df.loc[idx, 'study_id'])
        dic_case[df.loc[idx, 'study_id']].append(df.loc[idx, 'laterality'] + df.loc[idx, 'view_position'])
        dic_case[df.loc[idx, 'study_id']].append(df.loc[idx, 'breast_density'])
        dic_case[df.loc[idx, 'study_id']].append(df.loc[idx, 'breast_birads'])
        dic_case[df.loc[idx, 'study_id']].append(dic_groundtruth[int(df.loc[idx, 'breast_birads'].split(' ')[-1])])
        dic_case[df.loc[idx, 'study_id']].append(df.loc[idx, 'study_id'])
    '''    
    '''    if not pd.isnull(df.loc[idx, 'ShortPath']):
            #print(df.loc[idx, 'image_id'])
            #print(df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[-1].split('.')[0])
            if df.loc[idx, 'image_id']!=df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[-1].split('.')[0]:
                #print("original name:", output_folder_path_16bit+'/'+df.loc[idx, 'ShortPath'])
                #print("final name:", output_folder_path_16bit+'/'+df.loc[idx, 'ShortPath'].split('/')[0]+'/'+df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[0]+'_'+df.loc[idx, 'image_id']+'.png')
                os.rename(output_folder_path_16bit+'/'+df.loc[idx, 'ShortPath'], output_folder_path_16bit+'/'+df.loc[idx, 'ShortPath'].split('/')[0]+'/'+df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[0]+'_'+df.loc[idx, 'image_id']+'.png')
    '''
    
    #df_out = pd.DataFrame.from_dict(dic_case, orient='index', columns=['study_id', 'Views', 'BreastDensity', 'BIRADS', 'Groundtruth', 'ShortPath'])
    df_case.to_csv('/groups/dso/spathak/vindr/MG_training_files_vindr_multiinstance_groundtruth.csv', sep=';', na_rep='NULL', index = False)

def correcting_imagenames(path_to_input_csvfile, output_folder_path):
    df = pd.read_csv(path_to_input_csvfile, sep = ';')
    # open the file in the write mode
    for idx in df.index:
        if not pd.isnull(df.loc[idx, 'ShortPath']):
            #print(df.loc[idx, 'image_id'])
            #print(df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[-1].split('.')[0])
            if df.loc[idx, 'image_id']!=df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[-1].split('.')[0]:
                print("original name:", output_folder_path+'/'+df.loc[idx, 'ShortPath'])
                print("final name:", output_folder_path+'/'+df.loc[idx, 'ShortPath'].split('/')[0]+'/'+df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[0]+'_'+df.loc[idx, 'image_id']+'.png')
                os.rename(output_folder_path+'/'+df.loc[idx, 'ShortPath'], output_folder_path+'/'+df.loc[idx, 'ShortPath'].split('/')[0]+'/'+df.loc[idx, 'ShortPath'].split('/')[-1].split('_')[0]+'_'+df.loc[idx, 'image_id']+'.png')

def statistics(path_to_input_csvfile):
    df = pd.read_excel(path_to_input_csvfile)
    print("number of entries:", df.shape[0])
    print("groupby id:", df.groupby(by='ID1').size())


    #dic_groundtruth = {1:'benign', 2:'benign', 3:'benign', 4:'malignant', 5:'malignant'}

    #print(df.groupby(by='breast_birads').size())
    #df1 = df.groupby(by=['breast_birads','series_id']).size().reset_index(name='count')
    '''df['Views'] = df['laterality'] + df['view_position']
    df_series = df.groupby(by=['study_id','laterality'], as_index=False, group_keys=False).apply(aggregating_patient_case_info, dic_groundtruth)
    print(df_series.groupby(by='MaxBirads').size())
    print(df_series.groupby(by='Groundtruth').size())
    df_series = df.groupby(by=['study_id'], as_index=False, group_keys=False).apply(aggregating_patient_case_info, dic_groundtruth)
    print(df_series.groupby(by='MaxBirads').size())
    print(df_series.groupby(by='Groundtruth').size())
    '''
    '''
    df = pd.read_csv('/groups/dso/spathak/vindr/physionet.org/files/vindr-mammo/1.0.0/finding_annotations.csv', sep=',')
    dic_groundtruth = {1:'benign', 2:'benign', 3:'benign', 4:'malignant', 5:'malignant'}
    print(df.groupby(by='breast_birads').size())
    #df1 = df.groupby(by=['breast_birads','series_id']).size().reset_index(name='count')
    df['Views'] = df['laterality'] + df['view_position']
    df_series = df.groupby(by=['study_id'], as_index=False, group_keys=False).apply(aggregating_patient_case_info, dic_groundtruth)
    print(df_series.groupby(by='MaxBirads').size())
    print(df_series.groupby(by='Groundtruth').size())
    '''
    #df = pd.read_csv('/groups/dso/spathak/vindr/MG_training_files_vindr_multiinstance_groundtruth.csv', sep=';')
    #dic_groundtruth = {1:'benign', 2:'benign', 3:'benign', 4:'malignant', 5:'malignant'}
    #print(df.groupby(by='Groundtruth').size())

def create_SIL_csv_file(path_to_input_csvfile):
    df = pd.read_csv(path_to_input_csvfile, sep = ';')
    dic_groundtruth = {1:'benign', 2:'benign', 3:'benign', 4:'malignant', 5:'malignant'}
    df = df.rename(columns={'study_id':'StudyInstanceUID', 'series_id': 'SeriesInstanceUID', 'image_id':'ImageName', 'split': 'Split'})
    df['Views'] = df['laterality'] + df['view_position']
    df['BreastDensity'] = df['breast_density'].str.split(' ').str[-1]
    df['BIRADS'] = df['breast_birads'].str.split(' ').str[-1]
    print(df['BIRADS'])
    df['ImageLabel'] = df['BIRADS'].astype(int).map(dic_groundtruth)
    df['CaseLabel'] = df.groupby(by='StudyInstanceUID')['BIRADS'].transform(lambda x: dic_groundtruth[x.values.astype(int).max()])
    df['ShortPath'] = df['StudyInstanceUID'] + '/' + df['Views'] + '_' + df['ImageName'] + '.png'
    df = df.drop(['laterality','view_position', 'height', 'width', 'breast_density', 'breast_birads'], axis=1)
    df = df.sort_values(by=['StudyInstanceUID', 'Views'])
    df.to_csv('/groups/dso/spathak/vindr/MG_training_files_vindr_singleinstance_groundtruth.csv', sep=';', na_rep='NULL', index=False)

def count_mixed_viewgroundtruth(grp):
    grp_agg = pd.DataFrame(columns=['StudyInstanceUID', 'Conflict', 'CaseLabel'], index=range(1))
    
    grp_agg['LabelLeft'] = np.unique(grp[grp['Views'].str[0]=='L']['ImageLabel'])
    grp_agg['ConflictLeft'] = len(np.unique(grp_agg['LabelLeft']))>1
    
    grp_agg['LabelRight'] = np.unique(grp[grp['Views'].str[0]=='R']['ImageLabel'])
    grp_agg['ConflictRight'] = len(np.unique(grp_agg['LabelRight']))>1
    
    grp_agg['Conflict'] = (grp_agg['ConflictLeft'].item()==True) or (grp_agg['ConflictRight'].item()==True)
    grp_agg['CaseLabel'] = np.unique(grp['CaseLabel'])
    grp_agg['StudyInstanceUID'] = np.unique(grp['StudyInstanceUID'])
    return grp_agg

def count_mixed_sidegroundtruth(grp):
    grp_agg = pd.DataFrame(columns=['StudyInstanceUID', 'Conflict', 'CaseLabel'], index=range(1))
    
    grp_agg['LabelLeft'] = np.unique(grp[grp['Views'].str[0]=='L']['ImageLabel'])
    if len(np.unique(grp_agg['LabelLeft']))>1:
        if 'malignant' in list(np.unique(grp_agg['LabelLeft'])):
            grp_agg['LabelLeft']='malignant'
    else:
        grp_agg['LabelLeft']=np.unique(grp_agg['LabelLeft'])
    
    grp_agg['LabelRight'] = np.unique(grp[grp['Views'].str[0]=='R']['ImageLabel'])
    if len(np.unique(grp_agg['LabelRight']))>1:
        if 'malignant' in list(np.unique(grp_agg['LabelRight'])):
            grp_agg['LabelRight']='malignant'
    else:
        grp_agg['LabelRight']=np.unique(grp_agg['LabelRight'])
    
    grp_agg['Conflict'] = (grp_agg['LabelLeft']!=grp_agg['LabelRight'])
    grp_agg['CaseLabel'] = np.unique(grp['CaseLabel'])
    grp_agg['StudyInstanceUID'] = np.unique(grp['StudyInstanceUID'])
    return grp_agg

def count_mixed_grountruth(grp):
    grp_agg = pd.DataFrame(columns=['StudyInstanceUID', 'Conflict', 'CaseLabel'], index=range(1))
    grp_agg['Conflict'] = len(np.unique(grp['ImageLabel']))>1
    grp_agg['CaseLabel'] = np.unique(grp['CaseLabel'])
    grp_agg['StudyInstanceUID'] = np.unique(grp['StudyInstanceUID'])
    return grp_agg

def mixed_groundtruth_caselevel(path_to_input_csvfile):
    df = pd.read_csv(path_to_input_csvfile, sep = ';')
    #df_case = df.groupby(by=['StudyInstanceUID'], as_index=False, group_keys=False).apply(count_mixed_grountruth)#lambda x: len(np.unique(x['ImageLabel']))>1)
    #df_case = df.groupby(by=['StudyInstanceUID'], as_index=False, group_keys=False).apply(count_mixed_sidegroundtruth)
    df_case = df.groupby(by=['StudyInstanceUID'], as_index=False, group_keys=False).apply(count_mixed_viewgroundtruth)
    #df_case.columns.values[1] = 'Conflict'
    print(df_case)
    print(df_case['Conflict'].value_counts())
    print(df_case[df_case['CaseLabel']=='malignant']['Conflict'].value_counts())
    
def read_dicom(path_to_dicom_file, path_to_input_csvfile):
    folders_list = os.listdir(path_to_dicom_file)
    df = pd.read_excel(path_to_input_csvfile)
    df_new = pd.DataFrame(columns = ['Patient_Id', 'Age', 'View', 'Abnormality', 'ImageLabel', 'ImageLabelWNormal', 'Subtype', 'OriImgPath'])
    view_dic = {'medio-lateral oblique':'MLO', 'cranio-caudal': 'CC'}
    for patientid in folders_list:
        print(patientid)
        image_list = glob.glob(path_to_dicom_file + '/' + patientid+'/**/*.dcm', recursive=True)
        #image_list = os.listdir(path_to_dicom_file + '/' + patientid)
        for image_path in image_list:
            #image_path = path_to_dicom_file + '/' + patientid + '/' + image
            ds = pydicom.dcmread(image_path) 
            breastside_view =  ds.ImageLaterality + view_dic[ds.ViewCodeSequence[0].CodeMeaning]
            df_patient = df[(df['LeftRight'] == ds.ImageLaterality) & (df['ID1'] == patientid)]
            image_path_png = path_to_original_png + '/' + patientid + '/' + image_path.split('/')[-1].split('.')[0] + '.png'
            if df_patient.empty:
                df_patient = df[df['ID1'] == patientid]
                print(df_patient.index.item())
                df_patient.loc[df_patient.index.item(), 'abnormality'] = None
                df_patient.loc[df_patient.index.item(), 'classification'] = 'Normal'
                df_patient = df_patient.assign(classification1=['Benign'])
                #df_patient['classification1'] = df_patient['classification1'].astype('str')
            else:
                df_patient["classification1"] = df_patient["classification"]
            image_details = {'Patient_Id': df_patient.loc[df_patient.index.item(), 'ID1'], 'Age': df_patient.loc[df_patient.index.item(), 'Age'], 'View': breastside_view, 'Abnormality': df_patient.loc[df_patient.index.item(), 'abnormality'], 'ImageLabel': df_patient.loc[df_patient.index.item(), 'classification1'], 'ImageLabelWNormal': df_patient.loc[df_patient.index.item(), 'classification'],'Subtype': df_patient.loc[df_patient.index.item(), 'subtype'], 'OriImgPath': image_path_png}
            df_new.loc[len(df_new)]=image_details
        #print(df_new)
    df_new = df_new.sort_values(by=['Patient_Id'])
    df_new.to_csv(path_to_output_csvfile,  sep=';', na_rep='NULL', index = False)

#output_folder_path_16bit = '/groups/dso/spathak/vindr/processed_png_16bit'
#output_folder_path_8bit = '/groups/dso/spathak/vindr/processed_png_8bit'
path_to_input_csvfile = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/CMMD_clinicaldata_revision.xlsx'
path_to_dicom_file = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/manifest-1616439774456/CMMD'
path_to_output_csvfile = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/cmmd_singleinstance_groundtruth.csv'
path_to_original_png = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/original_png_8bit'

#w_mean_dataset, w_std_dataset, h_mean_dataset, h_std_dataset = calculate_image_size(output_folder_path_16bit)
#print(w_mean_dataset, w_std_dataset, h_mean_dataset, h_std_dataset)

#sanity_check(output_folder_path_16bit)

#create_MIL_csv_file(path_to_input_csvfile)

#statistics(path_to_input_csvfile)
#read_dicom(path_to_dicom_file, path_to_input_csvfile)

create_SIL_csv_file(path_to_input_csvfile)

#correcting_imagenames(path_to_input_csvfile, output_folder_path_8bit)

#mixed_groundtruth_caselevel(path_to_input_csvfile)

