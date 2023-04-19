import pandas as pd
#from geopy.distance import geodesic

def get_Sec_from_datetime(dt):
    timeStamp=pd.to_datetime(dt,format="%m/%d/%Y %H:%M:%S")
    return timeStamp.timestamp()

# def geodistance(pointA, pointB):
#     return geodesic(pointA, pointB).meters


def process_gps_file(fname):#,Lambda=100): #meter

    # NAN is not handled for this file reading could lead to error

    df_raw=\
    pd.DataFrame(pd.read_csv(fname).reset_index().values,
                 columns=['lat','long','speed','altitude','time','tag'])\
                .drop(columns=['tag','altitude'])\
                .astype({"lat":float,"long":float,"speed":float,"time":str})

    #top:-->1) reading file 2) renaming columns 3) droping unnecessary columns 4) converting to desired types

    #df_raw['timestamp']=df_raw.time.apply(get_Sec_from_datetime) #building timeStamp from datetime
    
    return df_raw