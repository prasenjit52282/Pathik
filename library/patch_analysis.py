from map_processing import MapFeatExtractor, geodistance
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

data_folder= os.getcwd() + "/../Data/Two_W"

PATCH_LENGTH = 100.0
PATCH_DISTANCE = 40.0

def patch_analysis(lat, lon):
    pass


def analyse_path_dir(dir):
    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(dir+'/'+f)]
    for file in files:
        analyse_file(file)


#Can use displacement instead of distance for a patch
def analyse_file(file):
    print(data_folder)
    mapFeatExtractor = MapFeatExtractor(data_folder)
    center_points = []
    dataframe = pd.DataFrame(pd.read_csv(file).reset_index().values,
                columns=['lat','long','speed','altitude','time','tag'])\
            .drop(columns=['tag','altitude'])\
            .astype({"lat":float,"long":float,"speed":float,"time":str})
    
    prev_point = (0,0)
    cumulative_dist = PATCH_LENGTH/2      # patches after dist 50. 150....
    for index, row in dataframe.iterrows():
        
        if not index:
            prev_point = (row["lat"], row["long"])
            continue

        point = (row["lat"], row["long"])
        
        curr_dist = geodistance(point, prev_point)
        cumulative_dist += curr_dist
        prev_point = point
        # print("cumulative dist = ", cumulative_dist)
        
        if cumulative_dist > PATCH_DISTANCE:
            # print("patch found")
            center_left_dist = PATCH_DISTANCE - cumulative_dist + curr_dist
            center_right_dist = curr_dist - center_left_dist
            
            #Calculate center point (current point crossing patch length)
            center_lat = (center_right_dist*prev_point[0] + center_left_dist*point[0])/curr_dist
            center_long = (center_right_dist*prev_point[1] + center_left_dist*point[1])/curr_dist
            center = (center_lat, center_long)
            center_points.append(center)

            cumulative_dist = center_right_dist

    poi_percent_dict = {}
    for pt in center_points:
        patch_data = mapFeatExtractor.get_features_from_circular_patch(pt[0], pt[1], PATCH_LENGTH)
        features = patch_data
        for key, val in features.items():
            if not val:
                continue                                                            #Doubt
            
            if key not in poi_percent_dict.keys():
                poi_percent_dict[key] = 0
            poi_percent_dict[key] += val
        
        mapFeatExtractor.highlight_path(pt[0],pt[1], PATCH_LENGTH)
        break
           
    for key in poi_percent_dict.keys():
        poi_percent_dict[key] /= len(center_points)
    
    print(poi_percent_dict)
    plt.pie(poi_percent_dict.values(), labels=poi_percent_dict.keys())
    plt.axis('equal')
    plt.show()

    # Create the image and display it
    patch_img_arr = np.multiply(mapFeatExtractor.trail_mask,mapFeatExtractor.area)
    patch_img = Image.fromarray(patch_img_arr, "RGB")
    patch_img.show()

    # Save the image to the desktop
    # patch_img.save(image_path)

# extract_traffic_noise("/Users/ajay/Desktop/MTP/mtp-bikesense-app/Data/Two_W/2019/DATA_15_34_16/All/bike_SOUND_2019_02_11_15_34_19_032.wav")
analyse_file("/Users/ajay/Desktop/MTP/bikesense/Data/Two_W/2019/DATA_18_13_26/All/bike_GPS_2019_02_04_18_13_26_299.txt")