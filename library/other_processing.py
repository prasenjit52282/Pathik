import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.impute import SimpleImputer
from .audio_processing import MFCC_on_Array, get_intensity_from_array
from .constants import sr,speed_threshold


def geodistance(pointA, pointB):
    return geodesic(pointA, pointB).meters


def tag_Lambda_meter_patch(df_in,Lambda=100):
    # return df_in with lambda meter tags added!!!
    df=df_in.copy()
    latlongs=df[['lat','long']].values
    start_lat,start_long=latlongs[0,:]

    patch_mark_list=[]
    cum_dis_list=[] #ADDED
    
    dis=0 #distance upto Lambda meters then reset
    patch_name="patch_" #patch name
    patch_number=0      #starting patch_number

    for lat,long in latlongs:
        dis+=geodistance((start_lat,start_long),(lat,long)) #distance from prev point(cumulative)
        patch_mark_list.append(patch_name+str(patch_number)) #patch no is tagged
        cum_dis_list.append(dis) #ADDED

        if dis>=Lambda: #Lambda is a input for crop length
            patch_number+=1 #next patch number is generated
            dis=0           #dis is reset to zero for next patch

        start_lat,start_long=lat,long

    df[f'Lambda_{Lambda}']=patch_mark_list
    df[f'cum_dis_{Lambda}']=cum_dis_list #ADDED
    return df


def merge_files(df_gps,df_acc,df_mac,df_audio):
    #sr=8000 #sampling rate of audio  @Imported from constants
    #Merging GPS(df_raw) with ACC(df_acc) and then replacing nan in acc column with mean of the column
    df_gps_acc=pd.merge(left=df_gps,right=df_acc,how='left',on="time")  #left join as nan acc values will be replaced with avg val

    #filling nan positions with mean acc
    imp_z_acc=SimpleImputer(strategy="mean")
    df_gps_acc['z_oriented_acc']=imp_z_acc.fit_transform(df_gps_acc[['z_oriented_acc']]).ravel()

    #Merging GPS_ACC(df_gps_acc) with MAC(df_mac) and then replacing nan in MAC column with 0 # of MAC count
    df_gps_acc_mac=pd.merge(left=df_gps_acc,right=df_mac,how='left',on="time")

    #filling nan positions with 0 MAC count
    imp_mac=SimpleImputer(strategy = "constant",fill_value=0)
    df_gps_acc_mac['MAC']=imp_mac.fit_transform(df_gps_acc_mac[['MAC']]).ravel()

    #Merging GPS_ACC_MAC(df_gps_acc+mac) with AUDIO(df_audio) and then replacing nan in all audio column with mean
    df_gps_acc_mac_audio=pd.merge(left=df_gps_acc_mac,right=df_audio,how='left',on="time")

    #filling nan positions with mean of columns
    imp_audio=SimpleImputer(strategy = "mean")

    df_gps_acc_mac_audio[[f'amp_{i}' for i in range(sr)]]=\
    imp_audio.fit_transform(df_gps_acc_mac_audio[[f'amp_{i}' for i in range(sr)]])
    
    return df_gps_acc_mac_audio



###Processing ROWs from Patchs

#sr=8000 #sampling rate for us 8000hz @Imported from constants
audio_columns=[f"amp_{i}" for i in range(sr)]
data_columns=['lat', 'long', 'speed', 'time', 'z_oriented_acc', 'MAC']+audio_columns

def get_processed_rows_for(data,x): #x meter Lambda
    group=data.groupby(f"Lambda_{x}")

    process_rows=[]
    for i,g in group:
        process_rows.append(process_patch(g[data_columns+[f'cum_dis_{x}']],x)) #ADDED

    return pd.DataFrame(process_rows)

def process_patch(d,x): #x meter patchs are being processed
    
    start_time,end_time=d.time.values[[0,-1]]
    #ADDED
    duration=(pd.to_datetime(end_time,format="%m/%d/%Y %H:%M:%S")- \
              pd.to_datetime(start_time,format="%m/%d/%Y %H:%M:%S")).seconds
    #NUMERICAL lat,longs
    lat,long=d[['lat','long']].mean().values
    #ADDED
    patch_length=d[f'cum_dis_{x}'].values[-1]
    
    #ADDED
    speed=(patch_length/(duration+1e-7))
    #NUMERICAL SPEED
    #speed=x/(d.speed.size+1e-7) #eps is added for handling inf values  @Imported from constants
    #same--> speed=distance/(total_time-stop_time)
    #//(d.speed>speed_threshold).sum() will give (total_time-stop_time) speed_threshold=0.5m/sec is threshold for stopage
    
    #WIFI
    mac=d.MAC.sum()
    #RSI
    rms_acc=np.sqrt((d.z_oriented_acc ** 2).sum ()/d.z_oriented_acc.size)
    rsi=rms_acc/speed#(d.z_oriented_acc.size*rms_acc)/d.speed.sum()
    #AUDIO
    audio_data = d[audio_columns].values.flatten()
    mfccs = MFCC_on_Array(audio_data)
    loudness = get_intensity_from_array(audio_data)

    #ADDED
    row=dict(zip(['lat','long','patch_length','duration','start_time','end_time',
                  'mac','rsi','mfcc0','mfcc1','mfcc2','mfcc3','mfcc4','speed', 'loudness'],
                 [lat,long,patch_length,duration,start_time,end_time,mac,rsi,*mfccs,speed, loudness]))
    return row