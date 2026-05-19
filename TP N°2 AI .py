# USHINDI MURHABAZI CHRISTOPHE
# RAISSA RUTEBUKA SCHOLASTIQUE
# TP N•2 D'AI
# Bac2 genie minier/ecole des mines/UOB

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données de diabète
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target 

# On garde 30% des données pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
#.......................
liste_modeles= {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=22),
    "Régression Linéaire": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=22)
}


print("      Résultats de l'Évaluation des Modèles demande dans la question        ")
for nom, liste_modele in liste_modeles.items():
    
    liste_modele.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = liste_modele.predict(X_test)
    
    #..........
    mse = round((mean_squared_error(y_test, y_pred)),3)
    r2 = round((r2_score(y_test, y_pred)),3)
    
    
    print("Modele",nom)
    print("MSE du modèle :", mse)
    print(" R²:",r2)
