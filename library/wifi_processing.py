import numpy as np
import pandas as pd
import re

#MAC finder pattern
mac_address_finder= re.compile(r'(?:[0-9a-fA-F]:?){12}')
#date time finder pattern
datetime_finder = re.compile(r'[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]') 

def find_mac(string):
    found=re.findall(mac_address_finder,string)
    #if found return else nan
    return found[0] if len(found)!=0 else np.nan

def find_datetime(string):
    found=re.findall(datetime_finder,string)
    #if found return else nan
    return found[0] if len(found)!=0 else np.nan


def process_wifi_file(fname):
    # NAN is handled for this file reading should not lead to error

    wifi_list=[]
    wifi_devs=pd.read_csv(fname,sep="\t",header=None).values.flatten()

    for line in wifi_devs:
        mac=find_mac(line)
        date_time=find_datetime(line)
        wifi_list.append([mac,date_time])

    #here nans are dropped
    df_mac=pd.DataFrame(wifi_list,columns=['MAC','time']).dropna()\
                        .groupby("time")[["MAC"]].count()\
                        .reset_index()

    #top--  1)wifi devs at each time is made as dataframe the nan are dropped(garbage case) 
    #       2) grouped by time and mac counts are taken
    #       3) reset ot get time as column
    return df_mac