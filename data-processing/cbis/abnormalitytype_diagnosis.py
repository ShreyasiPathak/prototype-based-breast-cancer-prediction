import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

pd.set_option('display.max_rows',None)

df = pd.read_csv('C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/raw dataset/cbis-ddsm/MG_training_files_cbis-ddsm_roi_groundtruth.csv', sep=';')
mass_groups = df.groupby(by=['AbnormalityType', 'MassShape', 'MassMargins'])
calc_groups = df.groupby(by=['AbnormalityType', 'CalcType', 'CalcDistribution'])
print(mass_groups.ngroups) #72
print(calc_groups.ngroups) #60
#print(df.groupby(by=['MassShape','MassMargins','Groundtruth','Assessment']).size())
#print(df.groupby(by=['MassMargins','Groundtruth','Assessment']).size())

mass_groups = df.groupby(by=['AbnormalityType', 'MassShape', 'MassMargins', 'Groundtruth'])
#print(mass_groups.size())
calc_groups = df.groupby(by=['AbnormalityType', 'CalcType', 'CalcDistribution', 'Groundtruth'])
#print(mass_groups.size())

# -------------------- mass group ---------------------------#
dic_descrip = dict()
for k, gp in mass_groups:
    #print(k, gp.shape[0])
    if 'mass'+'-'+k[1]+'-'+k[2] not in dic_descrip.keys():
        dic_descrip['mass'+'-'+k[1]+'-'+k[2]] = dict()
    if k[3] == 'malignant':
        dic_descrip['mass'+'-'+k[1]+'-'+k[2]]['malignant']=gp.shape[0]
    elif k[3] == 'benign':
        dic_descrip['mass'+'-'+k[1]+'-'+k[2]]['benign']=gp.shape[0]
    
#print(dic_descrip)

'''dic_descrip_both = dict()
for key in dic_descrip.keys():
    if len(dic_descrip[key].keys())>1:
        if (dic_descrip[key]['malignant']>15) or (dic_descrip[key]['benign']>15):
            dic_descrip_both[key] = dic_descrip[key]
print(dic_descrip_both)
'''

# ----------------------- calc group --------------------------#
for k, gp in calc_groups:
    #print(k, gp.shape[0])
    if 'calcification'+'-'+k[1]+'-'+k[2] not in dic_descrip.keys():
        dic_descrip['calcification'+'-'+k[1]+'-'+k[2]] = dict()
    if k[3] == 'malignant':
        dic_descrip['calcification'+'-'+k[1]+'-'+k[2]]['malignant']=gp.shape[0]
    elif k[3] == 'benign':
        dic_descrip['calcification'+'-'+k[1]+'-'+k[2]]['benign']=gp.shape[0]
    
print(dic_descrip)

dic_descrip_both = dict()
for key in dic_descrip.keys():
    if len(dic_descrip[key].keys())>1:
        #if (dic_descrip[key]['malignant']>15) or (dic_descrip[key]['benign']>15):
        #dic_descrip_both[key] = dic_descrip[key]
        dic_descrip_both[key] = dic_descrip[key].values()
print(dic_descrip_both)

df_dic_descrip_both = pd.DataFrame.from_dict(dic_descrip_both, orient='index', columns=['Benign', 'Malignant'])
df_dic_descrip_both.to_csv('./cbisddsm_abnormalitygroup_malignant_benign_count.csv', sep=';',na_rep='NULL',index=True)

#-------------------- plot -------------------------#
# set width of bar 
'''barWidth = 0.25
fig, ax = plt.subplots(figsize =(15, 15)) 
#fig, ax = plt.subplots(layout="constrained")

# set height of bar 
y_malignant = []
y_benign = []
for key in dic_descrip_both.keys():
    y_malignant.append(dic_descrip_both[key]['malignant'])
    y_benign.append(dic_descrip_both[key]['benign'])
 
# Set position of bar on X axis 
#br1 = np.arange(len(y_malignant)) 
br1 = np.arange(len(y_malignant)) 
br2 = np.array([x + barWidth for x in br1]) 

# Make the plot
ax.bar(br1, y_malignant, color ='r', width = barWidth, edgecolor ='grey', label ='malignant') 
ax.bar(br2, y_benign, color ='b', width = barWidth, edgecolor ='grey', label ='benign') 
 
# Adding Xticks 
plt.xlabel('Categories', fontweight ='bold', fontsize = 15) 
plt.ylabel('Count', fontweight ='bold', fontsize = 15) 
print(len(dic_descrip_both.keys()))
print(len(br1))
print(br1+barWidth)
print(list(dic_descrip_both.keys()))
#xlabels = ['ARCHDIS-SPICULATED', 'IRREGULAR-ILLDEFINED', 'IRREGULAR-SPICULATED', 'IRREGULAR-ARCHDIS-SPICULATED', 'LOBULATED-CIRCUMSCRIBED', 'LOBULATED-ILLDEFINED', 'LOBULATED-MICROLOBULATED', 'LOBULATED-OBSCURED', 'OVAL-CIRCUMSCRIBED', 'OVAL-ILLDEFINED', 'OVAL-MICROLOBULATED', 'OVAL-OBSCURED', 'OVAL-SPICULATED', 'ROUND-CIRCUMSCRIBED', 'ROUND-ILLDEFINED', 'ROUND-OBSCURED', 'ROUND-SPICULATED']
xlabels = ['AMORPHOUS-CLUSTERED', 'AMORPHOUS-SEGMENTAL', 'FINELINEARBRANCH-CLUSTERED', 'FINELINEARBRANCH-LINEAR', 'PLEOMORPHIC-CLUSTERED', 'PLEOMORPHIC-LINEAR', 'PLEOMORPHIC-REGIONAL', 'PLEOMORPHIC-SEGMENTAL', 'PUNCTATE-CLUSTERED', 'PUNCTATE-SEGMENTAL', 'PUNCTATE-PLEOMORPHIC-CLUSTERED']
ax.set_xticks(br1+(barWidth/2), xlabels, rotation = 35, ha='right', rotation_mode='anchor')
 
plt.legend()
plt.show() 

#fig.savefig("./calc_category_malignant_benign"+'.pdf', format='pdf', bbox_inches='tight')
'''