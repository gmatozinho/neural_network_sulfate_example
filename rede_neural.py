from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn import preprocessing
import joblib

import numpy as np
import pandas as pd

#se usar google colab colocar 1
usar_google_colab =0

if usar_google_colab==1:
# se for usar google colab
  from google.colab import files
  import io
  uploaded = files.upload()
  data = io.BytesIO(uploaded['Input_Capacidade_de_Sulfeto.csv']) #Colocar somente o nome do arquivo e nao o caminho
  df = pd.read_csv("Input_Capacidade_de_Sulfeto.csv",sep=",",decimal=".") #Colocar somente o nome do arquivo e nao o caminho
  #,header=None) - Inserir dentro do parenteses do df caso queira que leia a primeira linha do arquivo
else:
  #Leitura do arquivo (OBS: Ao colocar o caminho, observe a raiz (C:\ ou D:\ ou F:\)
  df = pd.read_csv("Input_Capacidade_de_Sulfeto.csv",sep=",",decimal=".")


#converte matriz para dataframe e separa os dados X e Y
x_train = pd.DataFrame(df.iloc[:,:6]) # neste caso coleta os dados das 8 primeiras colunas que são os Inputs
y_train = pd.DataFrame(df.iloc[:,6:7]) # neste caso coleta os dados da ultima coluna que são os Targets


x_train

y_train

#realização da normalização do x
min_max_scaler_x = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_train_norm = min_max_scaler_x.fit_transform(x_train)
joblib.dump(min_max_scaler_x,"min_max_scaler_x.xlsx")

#realização da normalização do y
min_max_scaler_y = preprocessing.MinMaxScaler(feature_range=(-1,1))
y_train_norm = min_max_scaler_y.fit_transform(y_train)
joblib.dump(min_max_scaler_y,"min_max_scaler_y.xlsx")


['min_max_scaler_y']

if usar_google_colab==1:  
  files.download('min_max_scaler_x.xlsx')
  files.download('min_max_scaler_y.xlsx')

neurons = 12 # editar com o numero de neuronios para o modelo
epochs = 10000 # numero de epocas, editar caso necessario

NN = MLPRegressor(hidden_layer_sizes=(neurons), max_iter=(epochs),tol=0.0000001,activation ='tanh', verbose=True)
NN.fit(x_train_norm,y_train_norm)

print('--------')    
print('%Acerto: ', NN.score(x_train_norm,y_train_norm)*100)
print('Perdas: ', NN.loss_)

# VERIFICACAO DO R2 - INICIO --------------------

# Armazenando os resultados preditos pela rede
# Calcula do R2 (coeficiente de correlacao)
y_predicted_norm = NN.predict(x_train_norm)
y_predicted_norm = pd.DataFrame(y_predicted_norm)


#Realizando a desnormalização do Y
y_predict_unnorm = min_max_scaler_y.inverse_transform(y_predicted_norm)

print('--------')
print('output_Train:')
print(y_predict_unnorm)
print('--------')

print('R2: ', r2_score(y_train_norm,y_predicted_norm))

# VERIFICACAO DO R2 - FIM --------------------

# SALVAR A REDE - INICIO -----------------------------------
import pickle
with open('network_tanh-1a1.xlsx', 'wb+') as savefile:   # alterar 'nome_da_rede' caso queira renomear o arquivo
    pickle.dump(NN,savefile)                  # alterar NN, caso NN nao seja a variavel que armazena a rede

# A rede neural se encontrara na mesma pasta que o código (.py),
# nomeada como 'nome_da_rede' (sem extensao)

if usar_google_colab==1:  
  files.download('network_tanh-1a1.xlsx')

# SALVAR A REDE - FIM --------------------------------------

#Concatena X com o a variável prevista pelo modelo, válido para matrizes numpy
x_ypredicted_unnorm = np.concatenate((x_train, y_predict_unnorm), axis=1)
x_ypredicted_unnorm = pd.DataFrame(x_ypredicted_unnorm)

if usar_google_colab==1:
  x_ypredicted_unnorm.to_csv ("Input_Capacidade_de_Sulfeto.csv", index = None, header=True)
  files.download("Input_Capacidade_de_Sulfeto.csv")
else:
  export_csv = x_ypredicted_unnorm.to_csv ("Input_Capacidade_de_Sulfeto.csv", index = None, header=True)

print('--------')
print('Input and Output do TREINAMENTO/Predicao')
print(x_ypredicted_unnorm.head)


# EXTRAINDO PESOS DAS CAMADAS E DAS BIAS INICIO ------------------------

# Extraindo pesos das entradas para a camada escondida
print('--------')
print('pesos das entradas para a camada escondida')

print(NN.coefs_[0])			# (pesos camadas)
print('--------')
print('pesos das Bias da camada escondida')
print(NN.intercepts_[0])		# (pesos bias)

# Extraindo pesos da camada escondida para as saídas
print('--------')
print('pesos da camada escondida para a camada de saida')
print(NN.coefs_[1])			# (pesos camadas)
print('--------')
print('pesos das Bias da camada de saida')
print(NN.intercepts_[1])		# (pesos bias)


df1 = pd.DataFrame(NN.coefs_[0]) # transforma os dados em um data frame
df1.to_excel('pesoscamadas.xlsx') # exporta os dados para formato excel
#from google.colab import files
#files.download('pesoscamadas.xlsx')

df2 = pd.DataFrame(NN.intercepts_[0]) # transforma os dados em um data frame
df2.to_excel('biascamadas.xlsx') # exporta os dados para formato excel
#files.download('biascamadas.xlsx')

df3 = pd.DataFrame(NN.coefs_[1]) # transforma os dados em um data frame
df3.to_excel('pesoscamadasaida.xlsx') # exporta os dados para formato excel
#files.download('pesoscamadasaida.xlsx')

df4 = pd.DataFrame(NN.intercepts_[1]) # transforma os dados em um data frame
df4.to_excel('pesobiassaida.xlsx') # exporta os dados para formato excel
#files.download('pesobiassaida.xlsx')

df5 = pd.DataFrame(y_predicted_norm) # transforma os dados em um data frame
df5.to_excel('y_predicted_norm.xlsx') # exporta os dados para formato excel
#files.download('y_predicted_norm.xlsx')

df6 = pd.DataFrame(y_predict_unnorm) # transforma os dados em um data frame
df6.to_excel('y_predict_unnorm.xlsx') # exporta os dados para formato excel
#files.download('y_predict_unnorm.xlsx')

df7 = pd.DataFrame(x_ypredicted_unnorm) # transforma os dados em um data frame
df7.to_excel('x_ypredicted_unnorm.xlsx') # exporta os dados para formato excel
#files.download('x_ypredicted_unnorm.xlsx')

if usar_google_colab==1:
  from google.colab import files
  files.download('pesoscamadas.xlsx')
  files.download('biascamadas.xlsx')
  files.download('pesoscamadasaida.xlsx')
  files.download('pesobiassaida.xlsx')
  files.download('y_predicted_norm.xlsx')
  files.download('y_predict_unnorm.xlsx')
  files.download('x_ypredicted_unnorm.xlsx')
else:
  df1.to_excel(r'pesoscamadas.xlsx') # exporta os dados para formato excel
  df1.to_excel(r'biascamadas.xlsx') # exporta os dados para formato excel
  df1.to_excel(r'pesoscamadasaida.xlsx') # exporta os dados para formato excel
  df1.to_excel(r'pesobiassaida.xlsx') # exporta os dados para formato excel
  df1.to_excel(r'y_predicted_norm.xlsx') # exporta os dados para formato excel
  df1.to_excel(r'y_predict_unnorm.xlsx') # exporta os dados para formato excel
  df1.to_excel(r'x_ypredicted_unnorm.xlsx') # exporta os dados para formato excel
