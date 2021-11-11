import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as met



#Ucitavanje podataka
podaci=pd.read_csv("novi_podaci.txt")
del podaci['field5410']

#Uklanjanje instanci koje sadrze null vrednost
podaci.dropna(inplace=True)

#Izdvajanje ciljnog atributa
atributi=podaci.columns[:5408].tolist()
x=podaci[atributi]
y=podaci["field5409"]

#Podela podataka na trening i test skup
x_trening,x_test,y_trening,y_test=train_test_split(x,y,test_size=0.3,stratify=y)

#Postavljanje parametara
k_vrednost=4
p_vrednost=2
tezina='uniform'

#Pravljenje modela
knn=KNeighborsClassifier(n_neighbors=k_vrednost,p=p_vrednost,weights=tezina)
knn.fit(x_trening,y_trening)


#Informacije o primeni modela na test skup
y_pred=knn.predict(x_test)
print("Test")
matrica_konf=met.confusion_matrix(y_test,y_pred)
df_matrica_konf=pd.DataFrame(matrica_konf,index=knn.classes_,columns=knn.classes_)
print("Matrica konfuzije")
print(df_matrica_konf)


