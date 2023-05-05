import numpy as np
import pandas as pd
import seaborn as sns
from itertools import groupby
import matplotlib.pyplot as plt
from collections import defaultdict
from library.constants import MAXSPEED

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] =(9,6)
plt.rcParams["axes.labelweight"] = "bold"

df=pd.read_csv("./Trails/TW/processed_data_100.csv")
#removing route3 from the data (route4 will be treated as route3)
df=df[~(df.route=='route3')].reset_index()
#computing speed in km/hr and clipping with max speed
df['speed[km/h]']=(df.speed*3.6).clip(0,MAXSPEED)
#renaming route4 with route3
df['route']=df.route.map({'route1':'R1','route2':'R2','route4':'R3'})


#Q1. length of each route (R1 to R3)
routes=df.groupby(['route','trail_no'])['patch_length'].sum()

route_dict=defaultdict(lambda:[])
for i,v in routes.items():
    route_dict[i[0]].append(v)

data=[]
for r,v in route_dict.items():
    data.append({'route': r, 'length(95%)[km]': np.round(np.percentile(v,q=95)/1000,2), 'Trails': len(v), 
                 'Total[km]':np.round(np.round(np.percentile(v,q=95))*len(v)/1000,2)})

routes_df=pd.DataFrame(data)
routes_df.to_csv("./logs/data_collection/routes.csv",index=False)
print(routes_df)


#Q2. Number of Trails per Route and Per Vehicle
vehicle_dict=defaultdict(lambda:{})
for k,v in df.groupby(['vehicle','route'])['trail_no'].unique().items():
    vehicle_dict[k[0]][k[1]]=len(v)

route_lengths=routes_df.set_index('route').drop(columns=['Trails','Total[km]']).to_dict()

vehicle_df=pd.DataFrame(dict(vehicle_dict)).T.replace(np.nan, 0).astype('int32')

vehicle_lengths=[]
for vehicle in vehicle_df.index:
    total=0
    for k,v in vehicle_df.loc[vehicle].items():
        total+=v*route_lengths['length(95%)[km]'][k]
    vehicle_lengths.append(round(total,2))

vehicle_df['Total[km]']=vehicle_lengths
vehicle_df=vehicle_df.reset_index().set_axis(['vehicle','R1','R2','R3','Total[km]'],axis=1)
vehicle_df.to_csv("./logs/data_collection/vehicles.csv",index=False)
print(vehicle_df)


#Q3. Route wise speed distribution
ls=iter(['solid','dashed','dotted'])
for g,d in df.groupby('route')['speed[km/h]']:
    sns.kdeplot(d.values,fill=True,label=str.capitalize(g),cut=0,linewidth=3,linestyles=next(ls))
plt.xlabel('Speed (km/h)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("./logs/data_collection/speed_dist.png")
plt.close()


#Q4. time-of-day wise plot
def time_of_day(ts):
    hour=int(str(ts).split()[1].split(':')[0])
    if 6<=hour<9:
        return 'Early\nMorning' #EM
    elif 9<=hour<12:
        return 'Morning' #M
    elif 12<=hour<15:
        return 'Noon' #N
    else:
        return 'After\nNoon' #AN

data_tod=df.start_time.apply(time_of_day).value_counts().to_dict()
tod=list(data_tod.keys())
tod_count=list(data_tod.values())

colors = sns.color_palette(n_colors=4)
explode = (0.05, 0.05, 0.05, 0.05)

fig, ax = plt.subplots()
ax.pie(tod_count, colors=colors, labels=tod,
        autopct='%1.1f%%', pctdistance=0.78,radius=0.8,
        explode=explode,startangle=90)

centre_circle = plt.Circle((0, 0), 0.51, fc='white')
ax.add_artist(centre_circle)
fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
ax.axis('equal')
ax.margins(0, 0)

plt.savefig("./logs/data_collection/tod_pie.png", pad_inches=0.2,
    bbox_inches='tight',)
plt.close()

#Q5. Year-wise Month wise day-wise distance collected
def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]
    
def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1./df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index,level):
            lxpos = (pos + .5 * rpos)*scale
            if level==2:
                ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes,fontdict=dict(size=12))
            else:
                ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale , ypos)
        ypos -= .1


df['year']=df.start_time.apply(lambda e:e.split(' ')[0].split('/')[-1])
df['day']=df.start_time.apply(lambda e:e.split(' ')[0].split('/')[-2])
df['month']=df.start_time.apply(lambda e:e.split(' ')[0].split('/')[-3])

ymd = np.round(df.groupby(['year','month','day'])['patch_length'].sum()/1000,2)
weekends=[pd.to_datetime(f'{e[1]}/{e[2]}/{e[0]} 00:00:00',
                format="%m/%d/%Y %H:%M:%S").dayofweek in [0,1] for e in ymd.index]

colors=sns.color_palette()
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ymd.plot(kind='bar',width=0.7,ax=fig.gca(),color=[colors[1] if v==True else colors[0] for v in weekends])
labels = ['' for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
ax.set_xlabel('')

label_group_bar_table(ax, ymd)
fig.subplots_adjust(bottom=.1*ymd.index.nlevels)
plt.ylabel('Distance (km)')

plt.grid()
plt.tight_layout()
plt.savefig("./logs/data_collection/date_data_col.png")
plt.close()