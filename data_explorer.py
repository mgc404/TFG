import pandas as pd
from sklearn.utils import shuffle
from random import choice

def get_celebrities(n_celebrities:int):
    df1, df2 = get_max_aparicions(get_identities(), n_celebrities)
    #df.reset_index(inplace=True, drop=True)
    return df1, df2


def get_identities():
    identity_df = pd.read_csv("Imatges\CelebA\Anno\identity_CelebA.txt", sep=' ', header=None)
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
