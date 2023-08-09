import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import poisson
from sklearn.decomposition import PCA

# Carrega o arquivo Excel com os resultados da Lotofácil
caminho_arquivo = '/content/loto_facil_asloterias_ate_concurso_2852_sorteio (1).xlsx'
try:
    df = pd.read_excel(caminho_arquivo)
except FileNotFoundError:
    print(f"Arquivo '{caminho_arquivo}' não encontrado.")
    exit(1)

# Verifica se o DataFrame tem as colunas esperadas
colunas_esperadas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Bola7', 'Bola8', 'Bola9', 'Bola10', 'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15', 'Ganhou']
if not all(coluna in df.columns for coluna in colunas_esperadas):
    print("O DataFrame não contém todas as colunas esperadas.")
    exit(1)

# Análise de Frequência
frequencia_numeros = df.iloc[:, 2:17].stack().value_counts()
probabilidades = frequencia_numeros / len(df)

# Análise de Pares e Trios
pares_trios = {}
for idx, row in df.iterrows():
    numeros = row[colunas_esperadas[:15]]
    for i in range(len(numeros)):
        for j in range(i + 1, len(numeros)):
            par = tuple(sorted([numeros[i], numeros[j]]))
            if par in pares_trios:
                pares_trios[par] += 1
            else:
                pares_trios[par] = 1

# Distribuição de Números Pares e Ímpares
numeros_pares = df[df[colunas_esperadas[:15]] % 2 == 0].count()
numeros_impares = df[df[colunas_esperadas[:15]] % 2 != 0].count()

# Distribuição de Números Altos e Baixos
numeros_altos = df[df[colunas_esperadas[:15]] >= 14].count()
numeros_baixos = df[df[colunas_esperadas[:15]] <= 13].count()

# Análise de Sequências
sequencias = {}
for idx, row in df.iterrows():
    numeros = row[colunas_esperadas[:15]]
    seq_count = 0
    max_seq_count = 0
    for num in numeros:
        if num in sequencias:
            seq_count += 1
            max_seq_count = max(max_seq_count, seq_count)
        else:
            seq_count = 0
        sequencias[num] = max_seq_count

# Teste de Independência (utilizando a correlação entre resultados consecutivos)
correlacao_consecutivos = df[colunas_esperadas[:15]].corr()

# Teste de Uniformidade
uniformidade = df[colunas_esperadas[:15]].apply(lambda x: x.value_counts(normalize=True))

# Análise de Tendências Temporais
df['Data'] = pd.to_datetime(df['Data'])
df['Ano'] = df['Data'].dt.year
tendencias_temporais = df.groupby('Ano')[colunas_esperadas[:15]].mean()

# Análise de Desvios (comparando frequência esperada e observada de cada número)
frequencia_esperada = len(df) / 15
desvios = frequencia_numeros - frequencia_esperada

# Análise de Ciclos
df['Ciclo'] = [random.choice(['A', 'B', 'C']) for _ in range(len(df))]

# Análise de Soma e Média
df['Soma'] = df[colunas_esperadas[:15]].sum(axis=1)
df['Media'] = df[colunas_esperadas[:15]].mean(axis=1)

# Análise de Dígitos
digitos = {}
for idx, row in df.iterrows():
    numeros = row[colunas_esperadas[:15]]
    for num in numeros:
        for digito in str(num):
            if digito in digitos:
                digitos[digito] += 1
            else:
                digitos[digito] = 1

# Análise de Paridade
paridade_secoes = {}
for idx, row in df.iterrows():
    numeros = row[colunas_esperadas[:15]]
    secao = ''
    for i, num in enumerate(numeros, start=1):
        secao += 'P' if num % 2 == 0 else 'I'
        if i % 5 == 0:
            if secao in paridade_secoes:
                paridade_secoes[secao] += 1
            else:
                paridade_secoes[secao] = 1
            secao = ''

# Análise de Intervalos
intervalos = {}
for idx, row in df.iterrows():
    numeros = sorted(row[colunas_esperadas[:15]])
    for i in range(len(numeros) - 1):
        intervalo = numeros[i + 1] - numeros[i]
        if intervalo in intervalos:
            intervalos[intervalo] += 1
        else:
            intervalos[intervalo] = 1

# Análise de Números Quentes e Frios
numeros_quentes = frequencia_numeros.head(10)
numeros_frios = frequencia_numeros.tail(10)

# Análise de Grupos de Números
grupos_numeros = {}
for idx, row in df.iterrows():
    numeros = row[colunas_esperadas[:15]]
    grupo = tuple(sorted(numeros[:5]))
    if grupo in grupos_numeros:
        grupos_numeros[grupo] += 1
    else:
        grupos_numeros[grupo] = 1

# Divide os dados em treinamento e teste
X = df[colunas_esperadas[:15]]
y = df['Ganhou']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Realiza previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcula a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy:.2f}')

# Modelo de Regressão Linear
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_linear = regressor.predict(X_test)

# Redes Neurais Artificiais
clf_neural = MLPClassifier(random_state=42, max_iter=1000)
clf_neural.fit(X_train, y_train)
y_pred_neural = clf_neural.predict(X_test)

# Modelo de Árvore de Decisão
clf_arvore = DecisionTreeClassifier(random_state=42)
clf_arvore.fit(X_train, y_train)
y_pred_arvore = clf_arvore.predict(X_test)

# Modelo de Machine Learning Ensemble
clf_ensemble = VotingClassifier(estimators=[('neural', clf_neural), ('tree', clf_arvore)], voting='soft')
clf_ensemble.fit(X_train, y_train)
y_pred_ensemble = clf_ensemble.predict(X_test)

# Modelo de Análise de Componentes Principais (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# Modelo de Distribuição de Poisson
def modelo_poisson(resultados_anteriores, numero):
    media_ocorrencias = resultados_anteriores.count(numero) / len(resultados_anteriores)
    probabilidade = poisson.pmf(k=1, mu=media_ocorrencias)
    return probabilidade

# Modelo de Caminhada Aleatória
def modelo_caminhada_aleatoria(resultados_anteriores, num_simulacoes):
    simulacoes = []
    for _ in range(num_simulacoes):
        simulacao = [random.choice(resultados_anteriores)]
        for _ in range(14):
            proximo_numero = random.choice(range(1, 26))
            simulacao.append(proximo_numero)
        simulacoes.append(simulacao)
    return simulacoes

# Exemplo de uso das funções de modelo
numero_para_prever = 10
probabilidade_poisson = modelo_poisson(df[colunas_esperadas[:15]].values.flatten(), numero_para_prever)

num_simulacoes = 5
simulacoes_caminhada = modelo_caminhada_aleatoria(df[colunas_esperadas[:15]].values.flatten(), num_simulacoes)

# Exemplo de uso das análises e modelos
print("Análise de Frequência:")
print(frequencia_numeros)

print("Análise de Pares e Trios:")
print(pares_trios)

print("Distribuição de Números Pares e Ímpares:")
print(numeros_pares)
print(numeros_impares)

print("Distribuição de Números Altos e Baixos:")
print(numeros_altos)
print(numeros_baixos)

print("Análise de Sequências:")
print(sequencias)

print("Teste de Independência (Correlação):")
print(correlacao_consecutivos)

print("Teste de Uniformidade:")
print(uniformidade)

print("Análise de Tendências Temporais:")
print(tendencias_temporais)

print("Análise de Desvios:")
print(desvios)

print("Análise de Ciclos:")
print(df['Ciclo'])

print("Análise de Soma e Média:")
print(df['Soma'])
print(df['Media'])

print("Análise de Dígitos:")
print(digitos)

print("Análise de Paridade:")
print(paridade_secoes)

print("Análise de Intervalos:")
print(intervalos)

print("Análise de Números Quentes e Frios:")
print(numeros_quentes)
print(numeros_frios)

print("Análise de Grupos de Números:")
print(grupos_numeros)

# ... (outros modelos e análises) ...

print("Análises e modelos concluídos.")
