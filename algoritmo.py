import pandas as pd
#para exportar para front
import joblib
# Importando as bibliotecas necessarias
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score   
from sklearn.metrics import recall_score      
from sklearn.metrics import f1_score



df = pd.read_csv('jogo_da_velha_dataset.csv')


# Separando o conjunto de dados em treinamento (60%), validacao (20%) e teste (20%)
df_treino, df_temp = train_test_split(df, test_size=0.4, random_state=42)
df_val, df_teste = train_test_split(df_temp, test_size=0.5, random_state=42)

# Extrai a variavel alvo do conjunto de treinamento, validacao e teste
df_treino_target = df_treino['result'].copy()
df_val_target = df_val['result'].copy()
df_teste_target = df_teste['result'].copy()

# Remove a coluna 'result' dos conjuntos de treinamento, validacao e teste
df_treino = df_treino.drop(columns=['result'])
df_val = df_val.drop(columns=['result'])
df_teste = df_teste.drop(columns=['result'])

# Pre-processamento das features categoricas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preproc_completo = ColumnTransformer([
    ('numericos',   'passthrough',    []),
    ('categoricos', OneHotEncoder(),  ['atr1','atr2','atr3','atr4','atr5','atr6','atr7','atr8','atr9']),
    ], 
    sparse_threshold=0)

# Aplica o pre-processamento nos conjuntos de treinamento, validacao e teste
X_treino = preproc_completo.fit_transform(df_treino)
X_val = preproc_completo.transform(df_val)
X_teste = preproc_completo.transform(df_teste)

# Atribui as variaveis alvo dos conjuntos de treinamento, validacao e teste
Y_treino = df_treino_target.values
Y_val = df_val_target.values
Y_teste = df_teste_target.values

# Define a grade de parametros para a busca em grade
param_grid = [{'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13],
               'min_samples_leaf': [2,3,4,5,6,7,8,9]}]

# Inicializa os modelos
mlp_clf = MLPClassifier(hidden_layer_sizes=(30,15), max_iter=500)
knn_clf = KNeighborsClassifier(7)
arvore = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Realiza a busca em grade para encontrar os melhores parametros para a arvore de decisao
grid_search = GridSearchCV(arvore, param_grid)
grid_search.fit(X_treino, Y_treino)
arvore = grid_search.best_estimator_

# Treina os modelos
mlp_clf.fit(X_treino, Y_treino)
knn_clf.fit(X_treino, Y_treino)
random_forest.fit(X_treino, Y_treino)

# Avalia os modelos no conjunto de validacao
mlp_pred_val = mlp_clf.predict(X_val)
knn_pred_val = knn_clf.predict(X_val)
arvore_pred_val = arvore.predict(X_val)
random_forest_pred_val = random_forest.predict(X_val)

# Exibe as metricas de avaliacao no conjunto de validacao
print("Metricas de Avaliacao no Conjunto de Validacao:")
print("------------------------------------------------")
print("Rede Neural (MLP):")
print("   Acuracia:", accuracy_score(Y_val, mlp_pred_val))
print("   Precisao:", precision_score(Y_val, mlp_pred_val, average='macro'))
print("   Recall:", recall_score(Y_val, mlp_pred_val, average='macro'))
print("   F1-score:", f1_score(Y_val, mlp_pred_val, average='macro'))
print("------------------------------------------------")
print("k-NN:")
print("   Acuracia:", accuracy_score(Y_val, knn_pred_val))
print("   Precisao:", precision_score(Y_val, knn_pred_val, average='macro'))
print("   Recall:", recall_score(Y_val, knn_pred_val, average='macro'))
print("   F1-score:", f1_score(Y_val, knn_pred_val, average='macro'))
print("------------------------------------------------")
print("Arvore de Decisao:")
print("   Acuracia:", accuracy_score(Y_val, arvore_pred_val))
print("   Precisao:", precision_score(Y_val, arvore_pred_val, average='macro'))
print("   Recall:", recall_score(Y_val, arvore_pred_val, average='macro'))
print("   F1-score:", f1_score(Y_val, arvore_pred_val, average='macro'))
print("------------------------------------------------")
print("Random Forest:")
print("   Acuracia:", accuracy_score(Y_val, random_forest_pred_val))
print("   Precisao:", precision_score(Y_val, random_forest_pred_val, average='macro'))
print("   Recall:", recall_score(Y_val, random_forest_pred_val, average='macro'))
print("   F1-score:", f1_score(Y_val, random_forest_pred_val, average='macro'))

joblib.dump(knn_clf, 'knn_model.pkl')
print("Model loaded successfully.")