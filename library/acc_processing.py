import pandas as pd


def orient_zaxis(x,y,z):
    g=9.8
    if(abs(abs(z)-g)<abs(abs(y)-g) and abs(abs(z)-g)<abs(abs(x)-g)):
        #print("Z Normal")
        return z
    elif(abs(abs(x)-g)<abs(abs(z)-g) and abs(abs(x)-g)<abs(abs(y)-g)):
        #print("X=Z")
        return x
    elif(abs(abs(y)-g)<abs(abs(z)-g) and abs(abs(y)-g)<abs(abs(x)-g)):
        #print("Y=Z")
        return y 
    else:
        return z
    
    
def process_acc_file(fname):

    # NAN is not handled for this file reading could lead to error

    df_acc_raw=\
    pd.DataFrame(pd.read_csv(fname).reset_index().values,
                 columns=['x','y','z','time','tag'])\
                .drop(columns='tag')\
                .astype({"x":float,"y":float,"z":float,"time":str})\
                .groupby("time")[['x','y','z']]\
                .mean()\
                .reset_index()

    #top-- 1) Reading file 2) Column taging 3) drop unnecessary 4) dtype convert 5) groupby time & mean(now it will be 1 sec each)
    #      5) then reset index to get time as column

    df_acc_raw['z_oriented_acc']=df_acc_raw[['x','y','z']].apply(lambda e:orient_zaxis(e['x'],e['y'],e['z']),axis=1)
    df_acc=df_acc_raw.drop(columns=['x','y','z'])

    #top-- 1) z_acc is oriented and the unnecessary columns are dropped
    return df_acc