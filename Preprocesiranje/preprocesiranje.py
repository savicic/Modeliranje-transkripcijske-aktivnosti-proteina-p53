import pandas as pd

#Ucitavanje podataka
podaci=pd.read_csv("novi_podaci.txt")

#Broj instanci
br_instanci=podaci.shape[0]
print("Broj instanci: ",br_instanci)
print("\n")


#Broj instanci bez null vrednosti
br_instanci=podaci.count()
print("Broj instancibez null vrednosti: ")
print(br_instanci)
print("\n")