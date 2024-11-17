# Predição de Engajamento de Influenciadores no Instagram (Regressão Linear)

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
- *docs/relatório_final.pdf* - Documento técnico com a descrição detalhada do projeto e resultados.
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