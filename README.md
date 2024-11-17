[20:33, 16/11/2024] Xany: # Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Passo 1: Carregar o Dataset
# Ajustar o caminho para acessar o arquivo no diretório 'data'
dataset_path = "data/top_insta_influencers_data.csv"
df = pd.read_csv(dataset_path)
print("Visualização das primeiras linhas do dataset:")
print(df.head())

# Remover colunas textuais irrelevantes ('channel_info', 'country') que não contribuem para o modelo
text_columns = ['channel_info', 'country']
for col in text_columns:
    if col in df.columns:
        print(f"Removendo coluna de texto irrelevante: {col}")
        df = df.drop(columns=[col])

# Funções para converter strings como 'k', 'm', e '%' em valores numéricos
def convert_k_m_to_numeric(value):
    if isinstance(value, str):
        if 'k' in value:
            return float(value.replace('k', '')) * 1e3
        elif 'm' in value:
            return float(value.replace('m', '')) * 1e6
    return value

def convert_percentage_to_float(value):
    if isinstance(value, str) and '%' in value:
        return float(value.replace('%', '')) / 100
    return value

# Aplicar conversões em colunas relevantes
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(convert_k_m_to_numeric)
        df[col] = df[col].apply(convert_percentage_to_float)

# Garantir que todas as colunas estão no formato numérico e preencher valores ausentes
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())

# Passo 2: Análise Exploratória dos Dados
# Objetivo: Identificar correlações entre as variáveis e distribuições
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação entre Variáveis")
plt.show()

# Exibir a distribuição das variáveis para análise inicial
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Distribuição das Variáveis do Dataset")
plt.show()

# Gráficos de dispersão para visualizar relações com a variável alvo ('influence_score')
target_column = 'influence_score'
feature_columns = [col for col in df.columns if col != target_column]

for col in feature_columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df[target_column])
    plt.title(f"Relação entre {col} e Influence Score")
    plt.show()

# Passo 3: Seleção de Recursos
# Análise de correlação para identificar variáveis relevantes
correlation_threshold = 0.1
corr_matrix = df.corr()
relevant_features_corr = corr_matrix[target_column][abs(corr_matrix[target_column]) > correlation_threshold].index
print(f"Variáveis relevantes com base na correlação: {list(relevant_features_corr)}")

# Manter apenas colunas relevantes com base na correlação
df = df[relevant_features_corr]

# Seleção automática de recursos com SelectKBest
X = df.drop(columns=[target_column])
y = df[target_column]

k = min(5, X.shape[1])  # Selecionar até 5 variáveis ou menos, dependendo da disponibilidade
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support(indices=True)]
print(f"Variáveis selecionadas automaticamente pelo SelectKBest: {list(selected_features)}")

# Atualizar X com as variáveis selecionadas
X = X[selected_features]

# Passo 4: Divisão dos Dados e Normalização
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Passo 5: Construção do Modelo de Regressão Linear
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Avaliação do modelo Linear
y_pred = linear_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\nDesempenho do Modelo de Regressão Linear:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Regularização com Ridge e Lasso
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
print(f"\nRidge RMSE: {ridge_rmse}")

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
print(f"Lasso RMSE: {lasso_rmse}")

# Otimização de Hiperparâmetros
ridge_params = {'alpha': [0.1, 1.0, 10.0]}
lasso_params = {'alpha': [0.01, 0.1, 1.0]}

ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='neg_mean_squared_error', cv=5)
lasso_grid = GridSearchCV(Lasso(), lasso_params, scoring='neg_mean_squared_error', cv=5)

ridge_grid.fit(X_train_scaled, y_train)
lasso_grid.fit(X_train_scaled, y_train)

print(f"\nMelhor parâmetro para Ridge: {ridge_grid.best_params_}")
print(f"Melhor parâmetro para Lasso: {lasso_grid.best_params_}")

# Passo 6: Visualização dos Resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Valores Reais')
plt.plot(y_pred, label='Predições - Regressão Linear', alpha=0.7)
plt.plot(ridge_pred, label='Predições - Ridge', alpha=0.7)
plt.plot(lasso_pred, label='Predições - Lasso', alpha=0.7)
plt.legend()
plt.title("Comparação entre Valores Reais e Preditos")
plt.show()
[20:41, 16/11/2024] Xany: # Predição de Engajamento de Influenciadores no Instagram (Regressão Linear)

##### Desenvolvido por Jéssica Pereira da Silva e Rebecca Santana Santos

Este projeto analisa as taxas de engajamento dos principais influenciadores do Instagram utilizando Regressão Linear. O modelo explora as relações lineares entre as variáveis para identificar fatores influentes no engajamento. Inclui validação, otimização e documentação detalhada.

---

## 1. Nome do Projeto
Predição de Engajamento de Influenciadores no Instagram (Regressão Linear)

---

## 2. Descrição do Projeto
O objetivo deste projeto é implementar e avaliar o desempenho de um modelo de Regressão Linear para prever taxas de engajamento. A análise considera variáveis relevantes para entender como elas impactam no engajamento dos influenciadores.

O projeto aborda:
- Limpeza e preparação do conjunto de dados.
- Exploração de relações entre variáveis.
- Construção, otimização e avaliação do modelo preditivo.
- Documentação detalhada do processo.

---

## 3. Estrutura dos Arquivos
- *data/top_insta_influencers_data.csv* - Conjunto de dados contendo informações sobre influenciadores e métricas de engajamento.
- *notebooks/LINEAR_REGRESSION_INSTA.ipynb* - Notebook Jupyter com o código e análises do modelo de Regressão Linear.
- *reports/relatório_final.pdf* - Documento técnico com a descrição detalhada do projeto e resultados.
- *README.md* - Este arquivo, com informações e instruções do projeto.

---

## 4. Requisitos
Certifique-se de ter as seguintes dependências instaladas antes de executar o projeto:
- Python 3.8 ou superior.
- Bibliotecas: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn.

---

## 5. Como Executar
1. Clone o repositório:
   bash
   git clone https://github.com/jessica550/insta-engagement-prediction
   
   
2. Acesse o diretório do projeto:
   bash
   cd insta-engagement-prediction
   
   
3. Abra o notebook para visualizar e executar as análises:
   bash
   notebooks/LINEAR_REGRESSION_INSTA.ipynb
   

---

## 6. Tecnologias Utilizadas
- *Linguagem*: Python
- *Bibliotecas Principais*:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-Learn

---

## 7. Principais Métricas Utilizadas
As métricas utilizadas para avaliar o modelo incluem:
- *MSE (Erro Médio Quadrático)*: Mede o erro quadrático médio das predições.
- *RMSE (Raiz do Erro Médio Quadrático)*: Representa o erro na mesma escala dos dados.
- *MAE (Erro Absoluto Médio)*: Mede o erro absoluto médio entre as predições e os valores reais.

---

## 8. Resultados
- O modelo apresentou desempenho consistente com um RMSE médio de aproximadamente *12.74*.
- Regularizações com Ridge e Lasso foram aplicadas para melhorar a generalização do modelo.

---

## 9. Autores e Colaboradores
- *Jéssica Pereira da Silva* - Desenvolvedora principal e responsável pelo modelo de Regressão Linear e análise de dados.
- *Rebecca Santana Santos* - Colaboradora na otimização do modelo e elaboração do relatório.