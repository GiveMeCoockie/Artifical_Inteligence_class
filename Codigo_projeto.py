#Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets, linear_model

#Leitura do arquivo csv
concatenado = pd.read_csv(r"C:\Users\lucas\Documents\IA\Projeto\AMLWorkshop-master\Data\unlabeledfeatures.csv")
concatenado.head() #Mostrar o head da variável

#Deletando colunas não desejadas
concatenado = concatenado.drop(columns="datetime")  # delete column of date
concatenado = concatenado.drop(columns="model")
concatenado.describe().transpose() #transpor a matriz
concatenado_2 = concatenado #usado para manter a variável original segura
concatenado_2 = concatenado_2.drop(concatenado_2[concatenado_2.age < 2].index) #retirando as linhas cujas colunas age forem menores que 2
concatenado_2.head() #verificar como que ficou

#Concatenando as colunas de erros em uma só
concatenado_3 = concatenado_2
concatenado_3['error1count'] = np.where(concatenado_3['error2count'] !=0, 1, concatenado_3['error1count'])
concatenado_3['error1count'] = np.where(concatenado_3['error3count'] !=0, 1, concatenado_3['error1count'])
concatenado_3['error1count'] = np.where(concatenado_3['error4count'] !=0, 1, concatenado_3['error1count'])
concatenado_3['error1count'] = np.where(concatenado_3['error5count'] !=0, 1, concatenado_3['error1count'])
concatenado_3['error1count'] = np.where(concatenado_3['error1count'] !=0, 1, concatenado_3['error1count'])

#Treinando
mlp2 = MLPClassifier(hidden_layer_sizes=(25,20,15,10),max_iter=1000)
X4 = concatenado_3.drop('error1count',axis = 1)
y4 = concatenado_3['error1count']
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4)
mlp2.fit(X4_train,y4_train)
predictions_4 = mlp2.predict(X4_test)
print(confusion_matrix(y4_test,predictions_4))#Print da confusion matrix
print(classification_report(y4_test,predictions_4))#Algumas estatísticas da rede treinada

#Novo ML
concatenado_4 = concatenado_3
concatenado_4 = concatenado_4.drop(concatenado_4[concatenado_4.machineID > 3].index)#Diminuindo quantidade de dados
concatenado_4 = concatenado_4.drop(columns="machineID")
concatenado_4.head()
X5 = concatenado_4.drop('error1count',axis = 1)
y5 = concatenado_4['error1count']
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5)
mlp2.fit(X5_train,y5_train)
predictions_5 = mlp2.predict(X5_test)
print(confusion_matrix(y5_test,predictions_5))
print(classification_report(y5_test,predictions_5))

#Resultados
pd.DataFrame({'pred': predictions_5}).plot(figsize=(14,6))
pd.DataFrame({'Erros': y5_test}).plot(figsize=(18,10))
print('concatenado:',concatenado_4['error1count'].shape)
print('teste:',y5_test.shape)
print('treinamento:',y5_train.shape)
print('prediction:',predictions_5.shape)

#Exportando os arquivos para excel
df2 = pd.DataFrame(y5_test)
df2.to_excel(r"C:\Users\lucas\Documents\IA\Projeto\AMLWorkshop-master\Data\y5_test.xlsx")
df3 = pd.DataFrame(predictions_5)
df3.to_excel(r"C:\Users\lucas\Documents\IA\Projeto\AMLWorkshop-master\Data\predictions_5.xlsx")