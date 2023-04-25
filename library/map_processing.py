import glob
import numpy as np

from PIL import Image
import pandas as pd
Image.MAX_IMAGE_PIXELS=None
from geopy.distance import geodesic

################CONSTANTS OF THIS MODULE#################
POI={
    'human_made':  (255,235,59),
    'natural_land':(219,136,54),
    'high_way':    (219,54,54),
    'two_way':     (70,54,219),
    'one_way':     (244,75,137),
    'water':       (33,208,224),
    'park':        (0,255,0),
    'school':      (255,255,255),
    'medical':     (255,0,0),
    'other_poi':   (128,128,128)
}

cor_lat,cor_long=23.559819, 87.309379
c_lat,c_long=23.558957, 87.310663

pix2lat=(cor_lat-c_lat)/180
pix2long=(c_long-cor_long)/240


def geodistance(pointA, pointB):
    return geodesic(pointA, pointB).meters

def create_3channel_circular_mask(h, w, center=None, radius=None):# UAV circular patch from uav_range (h&w)

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return np.tile(np.expand_dims(mask.astype(np.uint8),axis=2),3)
    

def get_poi_percentage(image,total_area,poi,d=1):
    """
    calculate a perticular POI percentage in a image
    """
    r,g,b=POI[poi]
    target=np.abs(image-np.array([r,g,b]))<=d 
    #API gives +1,-1 difference than encoding so introduced a tollerence of d

    poi_map=np.logical_and(np.logical_and(target[:,:,0],target[:,:,1]),target[:,:,2])

    return poi_map.sum()/total_area
#.............................................................................................#

def get_poi_feat(image,total_area):
    """
    calculate all POI percentage and return a dict of all POIs and their percentages
    """
    poi_feat={}

    for poi in POI.keys():
        poi_feat[poi]=get_poi_percentage(image,total_area,poi)
        
    #Normalizing
    covered=sum(list(poi_feat.values()))
    for poi in POI.keys():
        poi_feat[poi]/=covered

    return poi_feat

def rgb2gray(rgb):
    
    gray_img = rgb.copy()
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_img[:,:,0] = gray
    gray_img[:,:,1] = gray
    gray_img[:,:,2] = gray

    return gray_img

###########################MAIN OBJECT #############################################################
class MapFeatExtractor:
    def __init__(self, data_folder):
        self.data_folder=data_folder
        self.area=np.array(Image.open(glob.glob(self.data_folder+"/*_area.png")[0]))
        self.trail_display_image = rgb2gray(self.area)
        self.trail_mask = np.full(shape=self.area.shape,fill_value=0,dtype=np.uint8)
        self.lat_long=pd.read_csv(glob.glob(self.data_folder+"/*_lat_long.csv")[0],index_col="Unnamed: 0").T.to_dict()

        #DERIVED CONSTANTS#################################################
        self.height_pix,self.width_pix,_=self.area.shape
        self.area_height=geodistance((self.lat_long["top_left"]["lat"],self.lat_long["top_left"]["long"]),
                                     (self.lat_long["bottom_right"]["lat"],self.lat_long["top_left"]["long"]))

        self.area_width=geodistance((self.lat_long["top_left"]["lat"],self.lat_long["top_left"]["long"]),
                                    (self.lat_long["top_left"]["lat"],self.lat_long["bottom_right"]["long"]))

        self.avg_pixcel_per_meter=np.mean([self.height_pix/self.area_height,self.width_pix/self.area_width])

    def get_features_from_circular_patch(self,lat,long,diameter):
        row_no=int(round((self.lat_long["top_left"]["lat"]-lat)/pix2lat))

        col_no=int(round((long-self.lat_long["top_left"]["long"])/pix2long))

        delta=int(round(self.avg_pixcel_per_meter*(diameter/2)))

        selected_area=self.area[row_no-delta:row_no+delta,col_no-delta:col_no+delta].copy()

        cir_channel=create_3channel_circular_mask(selected_area.shape[0],selected_area.shape[1])

        patch=np.multiply(cir_channel,selected_area) # diameter circular patch
        total_area=cir_channel[:,:,0].sum() #taking only one channel
        
        return get_poi_feat(patch,total_area)
    
    
    
    def get_features_from_rectangular_patch(self, lat, long, sideLength):    #Doubt total_area calculation
        row_no=int(round((self.lat_long["top_left"]["lat"]-lat)/pix2lat))

        col_no=int(round((long-self.lat_long["top_left"]["long"])/pix2long))

        delta=int(round(self.avg_pixcel_per_meter*(sideLength/2)))

        selected_patch=self.area[row_no-delta:row_no+delta,col_no-delta:col_no+delta].copy()

        total_area=selected_patch.shape[0]*selected_patch.shape[1] #taking only one channel
        return get_poi_feat(selected_patch,total_area)
    
    # def create_image_for_patch(self, gps_points, patch_length):
    #     mean_lat = gps_points['lat'].mean()
    #     mean_lon = gps_points['long'].mean()
    #     row_mid = int(round((self.lat_long["top_left"]["lat"] - mean_lat) / pix2lat))
    #     col_mid = int(round((mean_lon - self.lat_long["top_left"]["long"]) / pix2long))
        
    #     patch_pixels = int(round(self.avg_pixcel_per_meter * ((patch_length+30) / 2)))
    #     patch_mask = np.full(shape=(2*patch_pixels, 2*patch_pixels,3),fill_value=0,dtype=np.uint8)

    #     trail_point_rad = 20
    #     selected_area = self.area[row_mid-patch_pixels:row_mid+patch_pixels,col_mid-patch_pixels:col_mid+patch_pixels].copy()
    #     patch_img = Image.fromarray(selected_area, "RGB")
    #     patch_img.show()

    #     target = np.full(shape=selected_area.shape,fill_value=0,dtype=np.uint8)
    #     for poi in ["high_way", "two_way", "one_way"]:
    #         r,g,b=POI[poi]
    #         target=np.logical_or(np.abs(selected_area-np.array([r,g,b]))<=1, target)
        
    #     poi_map=np.logical_and(np.logical_and(target[:,:,0],target[:,:,1]),target[:,:,2])
    #     poi_map = np.repeat(poi_map[:, :, np.newaxis], 3, axis=2)
    #     target = np.multiply(selected_area, poi_map)
    #     patch_img_road = Image.fromarray(target, "RGB")
    #     patch_img_road.show()

    #     for index, row in gps_points.iterrows():
    #         lat = row["lat"]
    #         long = row["long"]
    #         row_no = int(round((self.lat_long["top_left"]["lat"] - lat) / pix2lat)) - (row_mid - patch_pixels)
    #         col_no = int(round((long - self.lat_long["top_left"]["long"]) / pix2long)) - (col_mid - patch_pixels)
            
    #         delta = int(round(self.avg_pixcel_per_meter * (trail_point_rad / 2)))

    #         # trail_patch = patch_mask[row_no-delta:row_no+delta, col_no-delta:col_no+delta]
    #         cir_channel=create_3channel_circular_mask(2*delta, 2*delta)
    #         patch_mask[row_no-delta:row_no+delta, col_no-delta:col_no+delta] = np.logical_or(cir_channel, patch_mask[row_no-delta:row_no+delta, col_no-delta:col_no+delta])

    #     road_trail_image = np.multiply(target,patch_mask)
    #     patch_img = Image.fromarray(road_trail_image, "RGB")
    #     patch_img.show()


    # def patch_road_trail_image(gps_data_patches):
    #     count =0
    #     for gps_points in gps_data_patches:
    #         self.create_image_for_patch(gps_points, data_is_processed_for)
    #         count+=1
    #         if count > 3: break

    def get_features_from_circular_patch_save_patch(self,lat,long,filepath,diameter):
        row_no=int(round((self.lat_long["top_left"]["lat"]-lat)/pix2lat))

        col_no=int(round((long-self.lat_long["top_left"]["long"])/pix2long))

        delta=int(round(self.avg_pixcel_per_meter*(diameter/2)))

        selected_area=self.area[row_no-delta:row_no+delta,col_no-delta:col_no+delta].copy()

        cir_channel=create_3channel_circular_mask(selected_area.shape[0],selected_area.shape[1])

        patch=np.multiply(cir_channel,selected_area) # diameter circular patch
        total_area=cir_channel[:,:,0].sum() #taking only one channel
        
        patch_img = Image.fromarray(patch, "RGB")
        patch_img.save(filepath)
        return get_poi_feat(patch,total_area)
    

    def add_map_features_to_processed_data(self,p_data,gps_data_patches,img_dir,data_is_processed_for=100): # default 100 meter
        map_feats=[]
        lat_longs=p_data[['lat','long']]
        filepaths = []
        count =0
        for lat,long in lat_longs.values:
            filepath = f"{img_dir}/img_{data_is_processed_for}_{count}.png"
            map_feats.append(self.get_features_from_circular_patch_save_patch(lat,long,filepath,data_is_processed_for))
            filepath = filepath.split("TW_DP/TW_DP")[1]
            filepaths.append(f".{filepath}")
            count += 1
        
        # patch_road_trail_image(gps_data_patches)
            
        map_data=pd.DataFrame(map_feats)
        img_path_data = pd.DataFrame(filepaths,columns=["patch_path"])
        whole_data=pd.concat([p_data,map_data,img_path_data],axis=1)
        return whole_data

    def highlight_path(self, lat, long, sideLength):
        row_no = int(round((self.lat_long["top_left"]["lat"] - lat) / pix2lat))
        col_no = int(round((long - self.lat_long["top_left"]["long"]) / pix2long))
        delta = int(round(self.avg_pixcel_per_meter * (sideLength / 2)))
        
        cir_channel=create_3channel_circular_mask(2*delta, 2*delta)

        self.trail_mask[row_no-delta:row_no+delta, col_no-delta:col_no+delta] = np.logical_or(cir_channel, self.trail_mask[row_no-delta:row_no+delta, col_no-delta:col_no+delta])