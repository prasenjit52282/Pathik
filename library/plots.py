import os
import glob
import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

np.random.seed(121)

plt.rcParams.update({'font.size': 16})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

file_path = "Trails/TW/processed_data_100.csv"
output_dir = "Output/"

data_split = {'Early\nMorning':[],'Morning':[],'Noon':[],'Afternoon':[]}
time_zone = ['Early\nMorning','Morning','Noon','Afternoon']

speed_rsi_dict = {'speed': np.array([]), 'rsi': np.array([])}

first_row = True
with open(file_path) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        hour = int(row[5].split(' ')[1].split(':')[0])
        time_split = int((hour -6) / 3)
        speed_rsi_dict['speed'] = np.append(speed_rsi_dict['speed'], float(row[13]))
        speed_rsi_dict['rsi'] = np.append(speed_rsi_dict['rsi'], float(row[7]))
        if time_split>3:
            continue

        data_split[time_zone[time_split]].append(3.6*float(row[13]))

def plot_timezone_speed():
    plt.boxplot(list(data_split.values()),showfliers=False,labels=time_zone,patch_artist=True,notch=True,boxprops=dict(facecolor='lightgray', color='k'))
    plt.ylim(0,80)
    plt.xlabel("Time of Day")
    plt.ylabel("Speed (km/hr)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir+"timezone_speed.png")
    #plt.show()
    plt.close()

def get_bin_count(norm_dist):
    q1 = norm_dist.quantile(0.25)
    q3 = norm_dist.quantile(0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (len(norm_dist) ** (1 / 3))
    bin_count = int(np.ceil((norm_dist.max() - norm_dist.min()) / bin_width)) 
    return bin_count                          

def plot_speed_rsi():

    df = pd.read_csv(file_path)
    df_new = df.sample(1000)[["speed","rsi"]].astype("float64")
    df_new["speed"] *= 3.6
    # print(df_new.shape)
    df_new = df_new[(np.abs(stats.zscore(df_new)) < 2).any(axis=1)]
    # print(df_new.shape)
    bin=np.max([get_bin_count(df_new["speed"]),get_bin_count(df_new["rsi"])])
    # bin=100
    g = sns.jointplot(data = df_new, x = "speed", y = "rsi", space=0, kind = "reg", marker = '+', scatter_kws = {'alpha':0.2}, line_kws = {'color':'k','linestyle':'dashed', 'lw':1.5}, marginal_kws=dict(bins=bin,fill=True))
    g.plot_joint(sns.kdeplot, color="orange", zorder=0, levels=6)
    # g.plot_marginals(sns.rugplot, color="r",height=-.05, clip_on=False)
    # plt.tight_layout()
    plt.grid()
    plt.xlabel("Speed (km/hr)")
    plt.ylabel("RSI")
    plt.ylim(0,4)
    plt.xticks(np.arange(0,100,20))
    plt.savefig(output_dir+"rsi_speed.png", bbox_inches='tight')
    #plt.show()
    plt.close()

def process_trail(data):
    n=10
    data['speed']=data.speed.apply(lambda e: np.nan if e>120 else e).ffill()
    max_speed = data["speed"].groupby(np.arange(len(data))//n).max()
    return (max_speed[:-1].tolist(), max_speed[1:].tolist())


def plot_rash_driving():
    df = pd.read_csv(file_path)
    df.speed *= 3.6
    trails = df.groupby("trail_no")
    past_speeds = []
    future_speeds = []
    for trail_no, trail_data in trails:
        past_speed, future_speed = process_trail(trail_data)
        past_speeds.extend(past_speed)
        future_speeds.extend(future_speed)
    
    sns.regplot(x=past_speeds, y=future_speeds, line_kws = {'color':'k','linestyle':'dashed', 'lw':2})
    plt.xlabel("Prev Max Speed (km/hr)")
    plt.ylabel("Next Max Speed (km/hr)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir+"past_future_speed.png", bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_speed_loudness():
    df = pd.read_csv(file_path)
    df_new = df[["speed","loudness"]].astype("float64")
    # df_new = df_new[(np.abs(stats.zscore(df_new)) < 2).all(axis=1)]
    df_new.speed *= 3.6
    df_new.speed=df_new.speed.apply(lambda e: np.nan if e>120 else e).ffill()
    df_new=df_new[df_new.loudness > 0].sample(1000)

    # Create the main plot
    fig, ax = plt.subplots()
    sns.regplot(x=df_new.speed, y=df_new.loudness, line_kws = {'color':'k','linestyle':'dashed', 'lw':2}, ax=ax)

    ax.set_xlabel("Speed (km/hr)")
    ax.set_ylabel("Loudness (dB)")
    ax.grid()
    
    mean_speed = round(np.mean(df_new.speed),2)
    # Create the inset plot
    inset_ax = fig.add_axes([0.68, 0.74, 0.18, 0.13]) # [left, bottom, width, height]
    sns.kdeplot(df_new.speed,ax=inset_ax,fill=True,color="Orange",alpha=0.4)
    inset_ax.set_xlabel("Speed Distribution",fontdict={'weight': 'normal','size': 11})
    inset_ax.set_ylabel("",fontdict={'weight': 'normal','size': 12})
    inset_ax.set_xticks([0, mean_speed,100])
    inset_ax.set_xticklabels([0,mean_speed,100],fontdict={'weight': 'normal','size': 10})
    inset_ax.set_yticklabels([],fontdict={'weight': 'normal','size': 1})
    inset_ax.axvline(mean_speed, color ='k', lw = 1.5, linestyle='dashed', alpha = 0.75)
    inset_ax.set_xlim([0, 120])
    # inset_ax.grid()

    # Save and show the plot
    plt.savefig(output_dir+"speed_loudness.png", bbox_inches='tight')
    #plt.show()
    plt.close()


def plot_poi_speed():
    df = pd.read_csv(file_path)
    df_new = df[["speed", "human_made","natural_land",
                 "high_way","two_way","one_way","water",
                 "park","school","medical","other_poi"]].astype("float64")
    
    df_new.speed *= 3.6
    df_new.speed=df_new.speed.apply(lambda e: np.nan if e>120 else e).ffill()
    df_new["speed_category"] = df_new.speed.apply(lambda e: "fast" if e>60 else "medium" if e>30 else "slow")

    print(df_new.speed_category)
    df_new = df_new.groupby("speed_category").mean()

    # print(df_grouped)

    df_new = df_new.drop(['speed'],axis=1)
    print(df_new)
    # Set the 'speed_category' column as the index
    # df_new.set_index("speed_category", inplace=True)

    # Define the colors for the stacked bars
    colors = [ "#3498db", "#9b59b6", "#2ecc71", "#e67e22", "#34495e",  "#95a5a6", "#e74c3c", 
             "#f1c40f", "#1abc9c", "#bdc3c7"]

    # Plot the stacked bar chart
    ax = df_new.plot.bar(stacked=True, figsize=(10, 6),rot=0,color=colors)

    # Add axis labels and title
    ax.set_xlabel("Speed Category", fontdict={'size': 12})
    ax.set_ylabel("Number of POIs",fontdict={'size': 12})
    ax.set_xticklabels(df_new.index,fontdict={'weight': 'normal','size': 11})
    ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontdict={'weight': 'normal','size': 11})
    ax.set_title("Percentage of POI by Speed Category",fontsize=12)
    ax.legend(title="POI Type", bbox_to_anchor=(1.0, 1.0), fontsize=11, title_fontsize=14)  # add a legend

    plt.tight_layout()
    plt.savefig(output_dir+"speed_poi.png")
    # Show the plot
    #plt.show()
    plt.close()

plot_timezone_speed()
plot_speed_rsi()
plot_rash_driving()
plot_speed_loudness()
plot_poi_speed()
