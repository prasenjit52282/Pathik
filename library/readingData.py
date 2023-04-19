import glob
from .gps_processing import process_gps_file
from .acc_processing import process_acc_file
from .wifi_processing import process_wifi_file
from .audio_processing import process_audio_file
from .other_processing import tag_Lambda_meter_patch,merge_files


def read_data_from_folder(folder):
    gps_file=glob.glob(folder+"/*/*_GPS_*")[0]
    acc_file=glob.glob(folder+"/*/*_ACC_*")[0]
    wifi_file=glob.glob(folder+"/*/*_WiFi_*")[0]
    audio_file=glob.glob(folder+"/*/*.wav")[0]

    df_gps=process_gps_file(gps_file)
    df_acc=process_acc_file(acc_file)
    df_mac=process_wifi_file(wifi_file)
    df_audio=process_audio_file(audio_file)

    # Merging
    df_merged=merge_files(df_gps,df_acc,df_mac,df_audio)

    #tagging for Lambda meter patchs
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=100) #100 meter patch
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=200)
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=300)
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=400)
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=500) #500 meter patch
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=1000)
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=1500)
    df_merged=tag_Lambda_meter_patch(df_merged,Lambda=2000)
    
    return df_merged