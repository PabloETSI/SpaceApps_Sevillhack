# -- coding: utf-8 --
# from sklearn.decomposition import PCA
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
import os.path
# import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from datetime import datetime, timedelta
import pandas
from collections import deque
#%%
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
plt.style.use('./presentation.mplstyle')
plt.close('all')
fig = plt.figure(figsize=(14, 6))
# fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
axs = [None] * 3
axs[0] = fig.add_axes([0.1, 0.1, 0.8, 0.3])
axs[1] = fig.add_axes([0.7, 0.5, 0.2, 0.4])
axs[2] = fig.add_axes([0.1, 0.5, 0.5, 0.4], projection='3d')

init_idx = 0
step = 1000
keys = []
for i in range(0, 50):
    keys.append(f"F{i}")

row = df.iloc[init_idx]
line, = axs[0].plot(row[keys], linewidth=3)
line.set_animated(True)
axs[0].set_xlim([0,49])
axs[0].set_ylim([-1,1000])
axs[0].grid('on')

xticks = np.arange(0, 50, 10)
xlabels = [f'{x}' for x in xticks]
axs[0].set_xticks(xticks, labels=xlabels)
yticks = np.arange(0,1000, 200)
axs[0].set_yticks(yticks)
axs[0].set_title('Raw solar wind flux vs. flow speed', color='#FFFFFF')
axs[0].set_xlabel('Flow speed [-]')
axs[0].set_ylabel('Solar wind flux [-]')

kpqueue = deque(30 *[2], 30)
kpqueue.append(row['Kp'])
barlabels = [f'{x}' for x in range(30)]
kpbar = axs[1].bar(np.arange(30), kpqueue, label=barlabels,\
                   width=0.2)
txt = axs[1].text(5, 8, str(row['Date']), style='italic',
        bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 10})
# axs[1].set_xticklabels([])
# axs[1].set_xlim([0.5,1.5])
axs[1].set_ylim([0,9])
axs[1].set_title('Ground Kp')
axs[1].set_ylabel('Kp [-]')
axs[1].yaxis.grid(True)

for i in range(30):
    kpbar[i].set_animated(True)
txt.set_animated(True)

def extract_magnetic(bs):
    bs = row[['Bx', 'By', 'Bz']]
    
    return np.array([[bs['Bx'], 0, 0], [0, bs['By'], 0], [0, 0, bs['Bz']],\
             [bs['Bx'], bs['By'], bs['Bz']]])

barr = extract_magnetic(row)
colors = ['red', 'green', 'blue', 'orange']
cero = [0]*4
bp = axs[2].quiver(cero, cero, cero,\
            barr[:,0], barr[:,1], barr[:,2], color=colors)#\
                # length=0.1, normalize=True)
axs[2].set_xlim([-30,30])
axs[2].set_ylim([-30,30])
axs[2].set_zlim([-30,30])
bp.set_animated(True)
#%%
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#         FancyArrowPatch.draw(self, renderer)
#%%
bplot = [None] * 4
for i in range(4):
    bplot[i] = axs[2].plot([0, barr[i,0]], [0, barr[i,1]], \
                zs=[0, barr[i,2]], linewidth=2,\
                color=colors[i])
#%%
def Kpcolor(Kp):
    if Kp <= 3:
        return '#00FF55'
    elif Kp <= 6:
        return '#FFEE00'
    elif Kp <= 8:
        return '#FF1100'
    else:
        return '#CC0000'
    
def update_animation(i):
    if step * i < df.shape[0]:
        row = df.iloc[i*step]
    line.set_ydata(row[keys])
    txt.set_text(str(row['Date']))
    kpqueue.append(row['Kp'])
    for i in range(30):
        kpbar[i].set_height(kpqueue[i])
        kpbar[i].set_color(Kpcolor(kpqueue[i]))
    barr = extract_magnetic(row)
    for i in range(4):
        bplot[i][0].set_data_3d([0, barr[i,0]], [0, barr[i,1]], \
                    [0, barr[i,2]])

    return line,txt,bplot[0][0],bplot[1][0],bplot[2][0],bplot[3][0],
    # kpbar[i] for i in range(30)],

ani = animation.FuncAnimation(
    fig=fig, func=update_animation, interval=100, blit=True, save_count = 1440)

# plt.show()

#%%
# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=30,
                                metadata=dict(artist='Sevillhack'),
                                bitrate=1800)
ani.save('temp.gif', writer=writer)