import numpy as np
import os
import os.path
from datetime import datetime, timedelta
import pandas
from collections import deque
#%%
print(f"CWD: {os.getcwd()}")
os.chdir('C:\\Users\\Usuario\\Desktop\\Segundo MII\\SpaceApps')
with open('kpdata-gfz.txt') as f:
    lines = f.readlines()

load_data = True
if(load_data):
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
    print("Loaded successfully, merging")
    df = all_data.merge(dKp)
else:
    if not 'df' in locals():
        print("Mal?")
        raise NameError("Falta df")

#%%
if not 'Kpconv' in locals():
    Kpconv = pandas.DataFrame(\
    {'dB': [0, 5, 10, 20, 40, 70, 120, 200, 330, 500], \
     'Kp': np.arange(10)})
#%%
df['Kp_DSCOVR'] = -1
# By = deque([-1]*180, 180)
# Bz = deque([-1]*180, 180)
dateprev = df['Date'].iloc[0]
extB = [1e4,1e4,-1e4,-1e4] # Bymin Bzmin Bymax Bzmax

for i in range(df.shape[0]):
    if df['Date'].iloc[i] - dateprev <= timedelta(hours=3):
        if ~np.any(np.isnan([df['By'].iloc[i], df['Bz'].iloc[i]])):
            extB[0] = np.min([df['By'].iloc[i], extB[0]])
            extB[1] = np.min([df['Bz'].iloc[i], extB[1]])
            extB[2] = np.max([df['By'].iloc[i], extB[2]])
            extB[3] = np.max([df['Bz'].iloc[i], extB[3]])
    else:
        maxvar = np.max([extB[2] - extB[0], extB[3] - extB[1]])
        idxKp = (Kpconv['dB']-maxvar).abs().argsort()[:1].item()
        if (maxvar < Kpconv['dB'].iloc[idxKp]).item():
            idxKp = np.max([0, idxKp-1])
           
        df['Kp_DSCOVR'].at[i-1] = Kpconv['Kp'].iloc[idxKp]
        if ~np.any(np.isnan([df['By'].iloc[i], df['Bz'].iloc[i]])):
            extB[0] = df['By'].iloc[i]
            extB[1] = df['Bz'].iloc[i]
            extB[2] = df['By'].iloc[i]
            extB[3] = df['Bz'].iloc[i]
        else:
            extB = [1e4,1e4,-1e4,-1e4]
            
        dateprev = df['Date'].iloc[i]
        
        
print("fin")
#%% 
dfKp = df[['Kp', 'Kp_DSCOVR']]
dfKp.to_csv('.\\ComputedKp.csv', index=False)