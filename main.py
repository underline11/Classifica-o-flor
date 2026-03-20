import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# 1. Carga dos dados
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = df[data.feature_names]
y = df["target"]

# 2. Divisão treino/teste (random_state garante que o resultado seja sempre o mesmo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinamento
modelo = LogisticRegression(max_iter=500)
modelo.fit(X_train, y_train)

# 4. Avaliação
pred = modelo.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"📊 Acurácia do Modelo: {acc * 100:.2f}%")
print("\n📋 Relatório de Classificação:\n", classification_report(y_test, pred, target_names=data.target_names))

# 5. Teste com nova entrada
entrada = pd.DataFrame([{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}])

previsao_id = modelo.predict(entrada)[0]
nome_flor = data.target_names[previsao_id]

print(f"🌸 Previsão para a nova entrada: {nome_flor.upper()}")
