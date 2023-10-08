# -- coding: utf-8 --

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Especifica la ruta al archivo CSV que deseas leer
archivo_csv = r".\dsc_fc_summed_spectra_2017_v01.csv"

# Utiliza pandas para leer el archivo CSV
data_frame = pd.read_csv(archivo_csv)

nombre_base = "Columna"
numero_de_columnas = 54
nombres_de_columnas = [nombre_base + str(i) for i in range(1, numero_de_columnas + 1)]
data_frame.columns = nombres_de_columnas

x = pd.DataFrame([data_frame["Columna2"], data_frame["Columna3"], data_frame["Columna4"]]).T

# Inicializa el escalador MinMaxScaler
scaler = MinMaxScaler()

# Normaliza los datos
x = scaler.fit_transform(x)

x=x[77398:127670]

#normalización


y = np.roll(x, -20, axis=0)  # Desplaza los datos hacia arriba en 30 pasos de tiempo



X = []
Y = []

# Crear las secuencias de entrada y las etiquetas
j=0
for i in range(0, len(x) - 720):
    X.append(x[i:i+720])
    Y.append(y[i:i+20])
   
    if i%1000==0:
        j = j+1
        print("Sigo vivo: " + str(j) + "%.")

X = np.array(X)
Y = np.array(Y)


# Dividir los datos en conjuntos de entrenamiento y prueba
print("Hola")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)
print("Adiós")


# Crear el modelo
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(720 * 3,)))
model.add(Dense(20*3, activation='relu'))

# Configurar el optimizador con el learning rate deseado (por ejemplo, 0.001)
custom_optimizer = Adam(learning_rate=0.0001)

# Compilar el modelo con el optimizador personalizado
model.compile(loss='mse', optimizer=custom_optimizer,metrics=['accuracy'])


# Resumen del modelo
model.summary()

# Entrenar el modelo con tus datos de entrada X y etiquetas Y
X_train = X_train.reshape(-1, 720 * 3)
Y_train = Y_train.reshape(-1, 20 * 3)
model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.2)

# Evaluar el modelo en datos de prueba
X_test = X_test.reshape(-1, 720 * 3)
Y_test = Y_test.reshape(-1, 20 * 3)
loss, accuracy = model.evaluate(X_test, Y_test)


predictions = model.predict(X_test)
MAE = np.mean(np.abs(predictions-Y_test))
print(f"Puntuación de precisión (MAE): {MAE}")

Y_test_res = Y_test.reshape(-1, 20, 3)
predictions_res = predictions.reshape(-1, 20, 3)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# plot the data
ax.plot([1,2,4,5]+list(range(8,21)),Y_test_res[1,[0,1,3,4]+list(range(7,20)),0], color='tab:blue')
ax.plot([1,2,4,5]+list(range(8,21)),predictions_res[1,[0,1,3,4]+list(range(7,20)),0], color='tab:orange')

# Seleción de vector
n = 0 
# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plot the data
ax.plot([1,2,3,4,7,8,9,10]+list(range(12,21)),Y_test_res[n,[0,1,2,3,6,7,8,9]+list(range(11,20)),1], color='tab:blue')
ax.plot([1,2,3,4,7,8,9,10]+list(range(12,21)),predictions_res[n,[0,1,2,3,6,7,8,9]+list(range(11,20)),1], color='tab:orange')
ax.set(xlim=(1,20),ylim=(0,0.5))
ax.set_xticks(range(2,21,2))
ax.grid(True,linewidth=0.2)
ax.set_xlabel("Tiempo a futuro (min)")
ax.set_ylabel("Componente Y del campo magnético")
fig.savefig(f'NN {n}.png',dpi=300)


# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# plot the data
ax.plot([1,2,4,5]+list(range(8,21)),np.array(Y_test_res[1,[0,1,3,4]+list(range(7,20)),0])-np.array(predictions_res[1,[0,1,3,4]+list(range(7,20)),0]), color='tab:blue')


# Especifica la ruta al archivo CSV que deseas leer
archivo_csv2 = r".\dsc_fc_summed_spectra_2023_v01.csv"

# Utiliza pandas para leer el archivo CSV
data_frame2 = pd.read_csv(archivo_csv2)
data_frame2.columns = nombres_de_columnas

x2 = pd.DataFrame([data_frame2["Columna2"], data_frame2["Columna3"], data_frame2["Columna4"]]).T
times2 = pd.DataFrame([data_frame2["Columna1"]])

# Normaliza los datos
x2 = scaler.fit_transform(x2)
x2 =x2[:12700]
times2 = times2.iloc[0,:12700]

y2 = np.roll(x2, -20, axis=0)  # Desplaza los datos hacia arriba en 30 pasos de tiempo

X2 = []
Y2 = []

# Crear las secuencias de entrada y las etiquetas
j=0
for i in range(0, len(x2) - 720):
    X2.append(x2[i:i+720])
    Y2.append(y2[i:i+20])
   
    if i%1000==0:
        j = j+1
        print("Sigo vivo: " + str(j) + "%.")

X2 = np.array(X2)
Y2 = np.array(Y2)

# Entrenar el modelo con tus datos de entrada X y etiquetas Y
X2 = X2.reshape(-1, 720 * 3)
Y2 = Y2.reshape(-1, 20 * 3)

predictions2 = model.predict(X2)
MAE2 = np.mean(np.abs(predictions2-Y2))
print(f"Puntuación de precisión (MAE): {MAE2}")

Y2_res = Y2.reshape(-1, 20, 3)
predictions2_res = predictions2.reshape(-1, 20, 3)
#%%
def update_animation(n):
    txt.set_text(times2[n])
    line.set_ydata(Y2_res[n,range(0,20),2])
    line2.set_ydata(predictions2_res[n,list(range(0,15))+list(range(16,20)),2])
    return line,line2,txt#,kpbar


# plot the data
fig = plt.figure(dpi=160)
ax = fig.add_subplot(1, 1, 1)
txt = ax.text(2,0.1,times2[0],bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 10})
line, = ax.plot(range(1,21),Y2_res[0,range(0,20),2], color='tab:blue', label = 'Real values')
line2, = ax.plot(list(range(1,16)) +list(range(17,21)),predictions2_res[0,list(range(0,15)) +list(range(16,20)),2], color='tab:orange', label = 'Predicted values')
txt.set_animated(True)
line.set_animated(True)
line2.set_animated(True)
ax.set(xlim=(1,20),ylim=(0.4,0.6))
ax.set_xticks(range(2,21,2))
ax.grid(True,linewidth=0.2)
ax.set_xlabel("Predicted time (min)")
ax.set_ylabel("Normalized Z magnetic field component")
ax.legend(['Real values','Predicted values'],loc='lower right')
#fig.savefig(f'NN2 {n}.png',dpi=300)
#for n in range(len(Y2_res))
#    # plot the data
#    ax.plot([1,2,3,4,7,8,9,10]+list(range(12,21)),Y2_res[n,[0,1,2,3,6,7,8,9]+list(range(11,20)),1], color='tab:blue')
#    ax.plot([1,2,3,4,7,8,9,10]+list(range(12,21)),predictions2_res[n,[0,1,2,3,6,7,8,9]+list(range(11,20)),1], color='tab:orange')
    

ani = animation.FuncAnimation(
    fig=fig, func=update_animation, interval=20, blit=True, save_count = 1440)

#%%
writer = animation.PillowWriter(fps=30,
                                metadata=dict(artist='Sevillhack'),
                                bitrate=1800)
ani.save('NN2023_2.gif', writer=writer)
