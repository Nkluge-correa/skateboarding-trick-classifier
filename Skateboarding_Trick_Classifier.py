# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.signal import savgol_filter

# ----------------------------------------------------------------------------------------#
# Load Data
# ----------------------------------------------------------------------------------------#

x = pd.read_excel(r'data\data_all.xlsx')
y = pd.read_excel(r'data\target_all.xlsx')

# ----------------------------------------------------------------------------------------#
# Dados para NParrays
# ----------------------------------------------------------------------------------------#

data = x.to_numpy()
target = y.to_numpy()
data = data.astype('float32')

# ----------------------------------------------------------------------------------------#
# Dados para PDataframes
# ----------------------------------------------------------------------------------------#

alvos = pd.DataFrame(target, columns =['Nollie', 'Nollie-Shov-It', 'Kickflip', 'Shov-It', 'Ollie'])
dados = pd.DataFrame(data)

print(
    'Comjunto de dados: cada linha(row) é um sinal (82 pontos de aceleração)',
    '\n',
    dados,
    '\n',
    'Alvos: cada linha é um alvo (one-hot-encoded) dentre 5 classes',
    '\n',
    alvos
)


# ----------------------------------------------------------------------------------------#
# Sinal Bruto
# ----------------------------------------------------------------------------------------#


plt.figure(figsize = (10,8))
plt.plot(data[366], 'b', data[367], 'r-', data[368], 'g')
#data[0], 'b', data[1], 'r-', data[2], 'g' for Nollie
#data[96], 'b', data[97], 'r-', data[98], 'g' for Nollie Shov It
#data[201], 'b', data[202], 'r-', data[203], 'g' for Kickflip
#data[291], 'b', data[292], 'r-', data[293], 'g' for Shove It
plt.xlabel('Time', fontsize=20)
plt.ylabel('G-force', fontsize=20)
plt.title('Sinal Bruto: Ollie (x, y, z)', fontsize=20)
plt.show()

# ----------------------------------------------------------------------------------------#
# Filtrando dados com Savitzky – Golay
# ----------------------------------------------------------------------------------------#

for i in range(len(data)):
  data[i] = savgol_filter(data[i], 15, 5) #11 = tamanho da janela, 5 = ordem do polinômio

plt.figure(figsize = (10,8))
plt.plot(data[366], 'b', data[367], 'r-', data[368], 'g')
#data[0], 'b', data[1], 'r-', data[2], 'g' for Nollie
#data[96], 'b', data[97], 'r-', data[98], 'g' for Nollie Shov It
#data[201], 'b', data[202], 'r-', data[203], 'g' for Kickflip
#data[291], 'b', data[292], 'r-', data[293], 'g' for Shove It
plt.xlabel('Time', fontsize=20)
plt.ylabel('G-force', fontsize=20)
plt.title('Sinal Filtrado: Ollie (x, y, z)', fontsize=20)
plt.show()

# ----------------------------------------------------------------------------------------#
# Train/Test Split & Normalização
# ----------------------------------------------------------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, shuffle = True) 

mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std


# ----------------------------------------------------------------------------------------#
# Modelo
# ----------------------------------------------------------------------------------------#
HIDDEN_LAYER_01 =  128
HIDDEN_LAYER_02 =  128
HIDDEN_LAYER_03 =  64
LEARNING_RATE = 0.0006 
L1 = 0.001 
L2 = 0.001 
DROPOUT_RATE = 0.8 
EPOCHS =   100
BATCH =  5
regularizer = tf.keras.regularizers.l1_l2(l1=L1, l2=L2)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(82,)),
                                    tf.keras.layers.Dense(HIDDEN_LAYER_01, kernel_regularizer=regularizer, activation='relu'),
                                    tf.keras.layers.Dense(HIDDEN_LAYER_02, kernel_regularizer=regularizer, activation='relu'),
                                    tf.keras.layers.Dense(HIDDEN_LAYER_03, kernel_regularizer=regularizer, activation='relu'),
                                    tf.keras.layers.Dropout(DROPOUT_RATE),
                                    tf.keras.layers.Dense(5, activation='softmax')])

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer = opt,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()

# ----------------------------------------------------------------------------------------#
# Loop de validação (k-fold Cross Validation)
# ----------------------------------------------------------------------------------------#

k = 4
num_val_samples = len(X_train) // k
num_epochs = EPOCHS
all_acc_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    partial_X_train = np.concatenate(
        [X_train[:i * num_val_samples],
        X_train[(i + 1) * num_val_samples:]],
        axis=0)

    partial_y_train = np.concatenate(
        [y_train[:i * num_val_samples],
        y_train[(i + 1) * num_val_samples:]],
        axis=0)

    history = model.fit(partial_X_train, partial_y_train,
                        validation_data=(val_data, val_targets),
                        epochs=EPOCHS, batch_size=BATCH, verbose=1)
    
    val_loss, val_acc = model.evaluate(val_data, val_targets, verbose=1)

    history_dict = history.history
    history_dict.keys()

    acc = history_dict['categorical_accuracy']
    val_acc = history_dict['val_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    acc_history = history.history['categorical_accuracy']
    all_acc_histories.append(acc_history)
average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]

'''
# ----------------------------------------------------------------------------------------#
# Gráficos de acurácia e loss por tempo(epoch)
# ----------------------------------------------------------------------------------------#
history_dict.keys()

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (10,8))
plt.plot(epochs, loss, 'bo:', label='Treinamento loss')
plt.plot(epochs, val_loss, 'rs-', label='Validação loss')
plt.title('Treinamento e Validação: loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend()
plt.show()

plt.figure(figsize = (10,8))
plt.plot(epochs, acc, 'bo:', label='Treinamento acc')
plt.plot(epochs, val_acc, 'rs-', label='Validação acc')
plt.title('Treinamento e Validação: acurácia', fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Acurácia',fontsize=20)
plt.legend()

plt.show()

'''
# ----------------------------------------------------------------------------------------#
# Treinamento e Testagem
# ----------------------------------------------------------------------------------------#

model.fit(X_train, y_train,
          epochs=20, batch_size=5, verbose=1)
test_loss_score, test_acc_score = model.evaluate(X_test, y_test)

print('Score de Perda em teste: ', test_loss_score, '.')
print('Performance em teste: ', test_acc_score, '.')



# ----------------------------------------------------------------------------------------#
# Avaliando o Modelo
# ----------------------------------------------------------------------------------------#

pd.set_option('display.float_format','{:.3f}'.format)
predictions = model.predict(X_test)
sample =  int(input('Escolha qualquer amostra (entre 0 e 92): '))
a = predictions[sample] #Distribuição de probabilidade
df = pd.DataFrame(a, index=['Nollie', 'Nollie-Shov-It', 'Kickflip', 'Shov-It', 'Ollie'],
                  columns=['Distribuição de Probabilidade'])
df = df.transpose()
df_2 = pd.DataFrame(y_test[sample], index=['Nollie', 'Nollie-Shov-It', 'Kickflip', 'Shov-It', 'Ollie'],
                  columns=['Verdadeira Distribuição     '])
df_2.transpose()

print(
    df,
    '\n',
    df_2,
    'Skateboarding_Trick_Classifier - by Nicholas Kluge (Masters dissertation in Electrical Engineering).'
)
