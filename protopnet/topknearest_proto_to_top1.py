import pandas as pd

topkinfo_df = pd.read_csv("/home/pathaks/protopnet/saved_models/cmmd/convnext_tiny_13/006/net_trained_best_8_8_nearest_train_protopnet/protopnet_cbis_topk.csv")

for index, row in topkinfo_df.iterrows():
    if index%10 == 0:
        print(index, row.values)
