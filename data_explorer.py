import pandas as pd
import shutil
from sklearn.utils import shuffle
from random import choice

def move_img(df:pd.DataFrame):
    img_dir_scr = "Imatges/CelebA/Img/img_align_celeba/img_align_celeba/"
    img_dir_dst = "main_img/"
    for i, row in df.iterrows():
        img_scr = img_dir_scr + row["Image_name"]
        img_dst = img_dir_dst + row["Image_name"]
        shutil.copyfile(img_scr, img_dst)

def get_dic_index(df, traget_col):
    df = df[traget_col].value_counts()
    dic_index = df.to_dict()
    i = 0
    for c in dic_index:
        dic_index[c] = i
        i += 1
    return dic_index
    
def refactor_identity(df, traget_col):
    d = get_dic_index(df, traget_col)
    df["Class"] = df.apply(lambda row: d[row[traget_col]], axis=1)
    return df

def get_celebrities(n_celebrities:int, refactor_class=False):
    df1, df2 = get_max_aparicions(get_identities(), n_celebrities)
    #df.reset_index(inplace=True, drop=True)
    if refactor_class:
        df1 = refactor_identity(df1, 'Identity')
    return df1, df2


def get_identities():
    identity_df = pd.read_csv("main_img\identity_CelebA.txt", sep=' ', header=None)
    identity_df = identity_df.rename(columns={0:"Image_name", 1:"Identity"})
    return identity_df


def get_max_aparicions(df:pd.DataFrame, n_celebrities:int):
    aparicions_df = df["Identity"].value_counts()
    aparicions_df = aparicions_df.reset_index()
    aparicions_df = aparicions_df.rename(columns={'Identity':'aparicions', 'index':'Identity'})
    aparicions_df = aparicions_df.iloc[:n_celebrities]
    return df[df['Identity'].isin(aparicions_df['Identity'])].reset_index(drop=True), df[~df['Identity'].isin(aparicions_df['Identity'])].reset_index(drop=True)

def get_tvt(df:pd.DataFrame, target_name:str, train_size:int):
    total_classes = df[target_name].nunique()
    n_df_col = df.shape[0]
    df = shuffle(df)
    train_df = df.iloc[:int(n_df_col*train_size)]
    val_df = df.iloc[int(n_df_col*train_size):int(n_df_col*( train_size + (1-train_size)/2 ))]
    test_df = df.iloc[int(n_df_col*( train_size + (1-train_size)/2 )):]
    while True:
        if train_df[target_name].nunique() == total_classes and val_df[target_name].nunique() == total_classes and test_df[target_name].nunique() == total_classes:
            return train_df, val_df, test_df

        df = shuffle(df)
        train_df = df.iloc[:int(n_df_col*train_size)]
        val_df = df.iloc[int(n_df_col*train_size):int(n_df_col*( train_size + (1-train_size)/2 ))]
        test_df = df.iloc[int(n_df_col*( train_size + (1-train_size)/2 )):]
        
def show_gb(gb:pd.DataFrame, n_gb=0, num_groups=0):
    count = 0
    for key, item in gb:
        if n_gb == 0:
            print(gb.get_group(key), "\n")
        else:
            print(gb.get_group(key).head(n_gb), "\n")
        count += 1
        if count == num_groups:
            break

def gbdf_to_list(gb_df:pd.DataFrame.groupby, target_list_elem:str, sort=False):
    ll = []
    for key, df in gb_df:
        sub_list = list(df[target_list_elem])
        if sort:
            sub_list = shuffle(sub_list)
        ll += sub_list
    return ll
