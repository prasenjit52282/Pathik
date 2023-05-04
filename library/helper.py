import os
import json
import glob
import pandas as pd
from .constants import data_folder,target_folder,over_write
from .map_processing import MapFeatExtractor
from .readingData import read_data_from_folder
from .other_processing import get_processed_rows_for


class POI_open:
    def __init__(self):
        with open(f"{data_folder}/global_dictionary.json") as f:
            self.open_poi_dict=json.load(f)
    
    def num_of_poi_open(self,time): #time in 24hrs format
        num_poi=0
        for k,l in self.open_poi_dict.items():
            for ranges in l:
                if time>=ranges["open"] and time<=ranges["close"]:
                    num_poi+=1
        return num_poi


#HELPERS
mfe=MapFeatExtractor(data_folder) #map feature extractor
poi_cal=POI_open() #PoI open calculator

#name output folders in this manner
def get_folder_name(folder_number,src_folder_path):
    return "_".join(src_folder_path.split("/")[-3:])+f"_{folder_number}"


#processing sensor & map data from a Lambda meter long patch
def whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,Lambda=100): #default 100 meter
    p_data, gps_points=get_processed_rows_for(data,Lambda) #processing sensor_data
    p_data['vehicle']=vehicle #adding vehicle type
    p_data['route']=route #adding route type
    return mfe.add_map_features_to_processed_data(p_data, gps_points, img_dir, Lambda) #adding map data to it


#check if any data file is not saved for a perticular folder
def check_if_data_need_to_be_process(folder_name):
    status=\
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_100.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_200.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_300.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_400.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_500.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_1000.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_1500.csv")) or \
    (not os.path.exists(f"{target_folder}/{folder_name}/DATA_2000.csv"))

    return status


def get_route_vehicle_type(folder):
    gps_file=glob.glob(folder+"/*/*_GPS_*")[0]
    route_file=glob.glob(folder+"/*/*.md")[0]
    vehicle=gps_file.split("/")[-1].split("_")[0].lower()
    vehicle='bike' if vehicle=='byke' else vehicle #mistake in labeling
    route=route_file.split("/")[-1].split('.')[0]
    return route,vehicle


def completely_process_one_folder(folder_number,folder_path):
    folder_name=get_folder_name(folder_number,folder_path)
    if(check_if_data_need_to_be_process(folder_name) or over_write):
        #either file doesnot exist or over-write is true then it will process!
        os.makedirs(target_folder+'/'+folder_name,exist_ok=True)
        img_dir = f"{target_folder}/{folder_name}/patches"
        os.makedirs(img_dir,exist_ok=True)

        data=read_data_from_folder(folder_path)
        route,vehicle=get_route_vehicle_type(folder_path)

        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,100).to_csv(f"{target_folder}/{folder_name}/DATA_100.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,200).to_csv(f"{target_folder}/{folder_name}/DATA_200.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,300).to_csv(f"{target_folder}/{folder_name}/DATA_300.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,400).to_csv(f"{target_folder}/{folder_name}/DATA_400.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,500).to_csv(f"{target_folder}/{folder_name}/DATA_500.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,1000).to_csv(f"{target_folder}/{folder_name}/DATA_1000.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,1500).to_csv(f"{target_folder}/{folder_name}/DATA_1500.csv",index=False)
        whole_data_from_raw_data_for_lambda_meter_patch_sensor_plus_map(route,vehicle,data,img_dir,2000).to_csv(f"{target_folder}/{folder_name}/DATA_2000.csv",index=False)
    

def read_file(i,fname):
    df=pd.read_csv(fname)
    df["Trail_type"]= "2018" if "2018" in fname else "2019"
    df['trail_no']=i
    #other calculations
    df['POI_open']=df.start_time.apply(lambda e:poi_cal.num_of_poi_open(e.split()[1][:-3]))
    df['DayOfWeek']=df.start_time.apply(lambda e:pd.to_datetime(e,format="%m/%d/%Y %H:%M:%S").dayofweek)
    df['DayOfMonth']=df.start_time.apply(lambda e:pd.to_datetime(e,format="%m/%d/%Y %H:%M:%S").day)
    return df



def merge_all_processed_trails_and_dump():
    files_100=glob.glob(target_folder+"/*/*_100.csv")
    files_200=glob.glob(target_folder+"/*/*_200.csv")
    files_300=glob.glob(target_folder+"/*/*_300.csv")
    files_400=glob.glob(target_folder+"/*/*_400.csv")
    files_500=glob.glob(target_folder+"/*/*_500.csv")
    files_1000=glob.glob(target_folder+"/*/*_1000.csv")
    files_1500=glob.glob(target_folder+"/*/*_1500.csv")
    files_2000=glob.glob(target_folder+"/*/*_2000.csv")

    df_100=pd.concat([read_file(i,f) for i,f in enumerate(files_100)],axis=0).reset_index(drop="index")
    df_200=pd.concat([read_file(i,f) for i,f in enumerate(files_200)],axis=0).reset_index(drop="index")
    df_300=pd.concat([read_file(i,f) for i,f in enumerate(files_300)],axis=0).reset_index(drop="index")
    df_400=pd.concat([read_file(i,f) for i,f in enumerate(files_400)],axis=0).reset_index(drop="index")
    df_500=pd.concat([read_file(i,f) for i,f in enumerate(files_500)],axis=0).reset_index(drop="index")
    df_1000=pd.concat([read_file(i,f) for i,f in enumerate(files_1000)],axis=0).reset_index(drop="index")
    df_1500=pd.concat([read_file(i,f) for i,f in enumerate(files_1500)],axis=0).reset_index(drop="index")
    df_2000=pd.concat([read_file(i,f) for i,f in enumerate(files_2000)],axis=0).reset_index(drop="index")

    df_100.to_csv(target_folder+"/processed_data_100.csv",index=False)
    df_200.to_csv(target_folder+"/processed_data_200.csv",index=False)
    df_300.to_csv(target_folder+"/processed_data_300.csv",index=False)
    df_400.to_csv(target_folder+"/processed_data_400.csv",index=False)
    df_500.to_csv(target_folder+"/processed_data_500.csv",index=False)
    df_1000.to_csv(target_folder+"/processed_data_1000.csv",index=False)
    df_1500.to_csv(target_folder+"/processed_data_1500.csv",index=False)
    df_2000.to_csv(target_folder+"/processed_data_2000.csv",index=False)
    
