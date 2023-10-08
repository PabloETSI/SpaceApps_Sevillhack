# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pandas
import os
import numpy as np

os.chdir('C:\\Users\\Usuario\\Desktop\\Segundo MII\\SpaceApps')
with open('kpdata-gfz.txt') as f:
    lines = f.readlines()

print("Loading Kp")
datetimes = []
Kps = []
for line in lines:
    l = line[:13].replace(' ','')
    lin = l[:8] + 'T' + l[8:]
    # datetimes.append(datetime.fromisoformat(lin))
    # Kps.append(float(line[47:52]))
    # Truco sucio
    for rep in range(0,180):
        datetimes.append(datetime.fromisoformat(lin) \
                         +timedelta(minutes=rep) \
                         )
        Kps.append(float(line[47:52]))
dicts = {'Date': datetimes, 'Kp': Kps}
dKp = pandas.DataFrame(data=dicts)
# df = pandas.DataFrame([datetimes, Kps])
# datez = np.datetime64(datetimes)
# datez = pandas.to_datetime(datetimes)

print("Loading DISCOVR data...\n")
colnames = ['Date', 'Bx', 'By', 'Bz']
for i in range(0, 50):
    colnames.append(f"F{i}")
data = []
# documents = glob('.\datos\*.csv')
os.chdir('C:\\Users\\Usuario\\Desktop\\Segundo MII\\SpaceApps')
folder = os.path.join(os.getcwd(), 'datos')
documents = os.listdir(folder)
for document in documents:
    data.append(pandas.read_csv(os.path.join(folder, document), \
    delimiter = ',', parse_dates=[0], names=colnames, \
    # na_values='0', \
    header = None))

all_data = pandas.concat(data)
print("Loaded successfully")

#%%
df = all_data.merge(dKp)
df.to_csv('.\\merged.csv', index=False)
#%%
# wr = all_data
# dt = all_data[0]
# # dt = all_data[0].to_numpy()
# j = 0
# for i in range(all_data.shape[0]):
#     while dt.iloc[i] >= datez.iloc[j]:
#         j += 1
        
#     wr[i].append(Kps[j])