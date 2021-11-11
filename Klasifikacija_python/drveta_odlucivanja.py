import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as met


def calculate_metrics(skup, tacne_vrednosti, predvidjene_vrednosti):

    print('Skup ',skup)

    #Pravljenje matrice konfuzije
    print('Matrica konfuzije')
    matrica_konfuzije=met.confusion_matrix(tacne_vrednosti,predvidjene_vrednosti)
    df_matrica_konf=pd.DataFrame(matrica_konfuzije,index=dt.classes_,columns=dt.classes_)
    print(df_matrica_konf)
    print('\n\n')

    #racunanje preciznosti
    preciznost=met.accuracy_score(tacne_vrednosti,predvidjene_vrednosti)
    print('Preciznost ',preciznost)

    preciznost = met.accuracy_score(tacne_vrednosti, predvidjene_vrednosti,normalize=False)
    print('Preciznost u broju instanci ', preciznost)
    print('\n\n')

    print('Preciznost po klasama ' ,met.precision_score(tacne_vrednosti,predvidjene_vrednosti, average=None))

    print('Odziv po klasama ', met.recall_score(tacne_vrednosti,predvidjene_vrednosti,average=None))


    #Pravljenje izvestaja klasifikacije
    izvestaj=met.classification_report(tacne_vrednosti,predvidjene_vrednosti)
    print('Izvestaj klasifikacije ',izvestaj, sep='\n')



#Ucitavanje podataka
podaci=pd.read_csv("novi_podaci.txt")



#Brisanje poslednje kolone jer je suvisna
del podaci['field5410']

#Uklanjanje instancikoje imaju null vrijednost
podaci.dropna(inplace=True)


atributi=podaci.columns[:5408].tolist()
x=podaci[atributi]
y=podaci["field5409"]

#Podela podataka na skup za treningi skup za test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,stratify=y)

#Praljenje i treniranje modela
dt=DecisionTreeClassifier(min_impurity_split=0.08,class_weight={'active':320,'inactive':7})
dt.fit(x_train,y_train)


#Primena modela na trening podacima
y_pred=dt.predict(x_train)
calculate_metrics('Trening',y_train,y_pred)

#Primena modela na test podacima
y_pred=dt.predict(x_test)
calculate_metrics('Test',y_test,y_pred)
