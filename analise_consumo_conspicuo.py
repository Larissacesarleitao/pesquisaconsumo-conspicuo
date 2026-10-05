"""
Consumo conspícuo: associações com fatores socioeconômicos e redes sociais
---------------------------------------------------------------------------
Código de análise do TCC (MBA em Data Science e Analytics - USP/Esalq).
Autora: Larissa Mendonça Cesar Leitão

O código segue a ordem do trabalho:
  Parte 1  Carga e limpeza da base, construtos e perfil da amostra (Tabelas 1 e 2)
  Parte 2  Confiabilidade e validade dos construtos (Tabelas 3 a 6)
  Parte 3  Correlações de Pearson e Spearman (Tabelas 7 e 8)
  Parte 4  Regressões simples e múltiplas, VIF e critério de Green (Tabelas 9 a 14)
  Parte 5  Figuras 1 e 2 e exportação das tabelas para o Excel
"""

# %% 0. Bibliotecas e configurações
import numpy as np
import pandas as pd

#obs: pandas: biblioteca para trabalhar com tabelas (DataFrames). Lê o Excel, filtra linhas, cria colunas, calcula médias e correlações.
#obs numpy: biblioteca que usa matemática com números e vetores (raiz quadrada, médias, valores ausentes np.nan). O pandas usa o numpy por baixo; no seu código ele aparece para o np.nan ("prefiro não responder") e, nas próximas partes, para np.sqrt.


# Para ajuste de exibição e que não ocorra quebra na visualização de dados.
pd.set_option("display.width", 160)   # o pandas imprime tabelas com até 160 caracteres de largura, antes de quebrar linha.
pd.set_option("display.max_columns", 50) # mostra até 50 colunas, em vez de esconder o meio com "...".

# A base original NÃO é disponibilizada, foi usada para fins acadêmicos (dados individuais que podem gerar reidentificação).
# Para rodar, coloque a planilha na mesma pasta deste código ou atualize abaixo com o caminho e arquivo entre "" 
CAMINHO_XLSX = "Pesquisa - Consumo Conspícuo.xlsx"
ABA = "Base Resultado Final"

# Dicionário abaixo onde as tabelas do trabalho vão sendo guardadas
# (na última parte, elas são exportadas para o Excel).
tabelas = {}


# %% 1. Carregar e limpar a base
df = pd.read_excel(CAMINHO_XLSX, sheet_name=ABA, header=0)
print("Formato original:", df.shape)

# A primeira coluna é o carimbo de data/hora do formulário:
# linhas sem carimbo são linhas em branco da planilha.
df = df[df.iloc[:, 0].notna()].copy()
print("Após remover linhas em branco:", df.shape)


# A segunda coluna pergunta se a pessoa tem 18 anos ou mais.
# Mantém só quem respondeu "Sim" (a pesquisa exclui menores de idade).
print(df.iloc[:, 1].value_counts(dropna=False))
df = df[df.iloc[:, 1] == "Sim"].copy()
print("N final (maiores de 18 anos):", len(df))   # esperado: 90


# %% 2. Identificar variáveis e construtos (por posição das colunas)
cols = df.columns.tolist()

# Itens de cada construto (escala Likert de 1 a 5)
ecs_items = cols[9:20]     # ECS: 11 itens
iprs_items = cols[22:27]   # IPRS: 5 itens
cc_items = cols[30:38]     # CC: 8 itens
print("Itens -> ECS:", len(ecs_items), "| IPRS:", len(iprs_items), "| CC:", len(cc_items))

# Renomeia as colunas usadas nas análises
df = df.rename(columns={
    cols[2]: "faixa_etaria",
    cols[3]: "genero",
    cols[4]: "escolaridade",
    cols[5]: "renda",
    cols[20]: "ECS",     # índice = média dos 11 itens de ECS
    cols[27]: "IPRS",    # índice = média dos 5 itens de IPRS
    cols[38]: "CC",      # índice = média dos 8 itens de CC
    cols[29]: "compra_influenciada_redes_raw",
    cols[40]: "compra_impressionar_raw",
})

# Conferência: os índices da planilha são mesmo a média simples dos itens?
for nome, itens in [("ECS", ecs_items), ("IPRS", iprs_items), ("CC", cc_items)]:
    media_itens = df[itens].apply(pd.to_numeric).mean(axis=1)
    dif = (df[nome].astype(float) - media_itens).abs().max()
    print(f"Conferência {nome}: maior diferença = {dif:.2e}")   # deve ser ~0

for nome in ["ECS", "IPRS", "CC"]:
    df[nome] = df[nome].astype(float)

# Comportamentos de compra declarados (Sim/Não -> 1/0)
df["compra_influenciada_redes"] = df["compra_influenciada_redes_raw"].map({"Sim": 1, "Não": 0})
df["compra_impressionar"] = df["compra_impressionar_raw"].map({"Sim": 1, "Não": 0})


# %% 3. Codificar as variáveis socioeconômicas
# Faixa etária, escolaridade e renda: ordinais, com valores crescentes.
# Posteriormente rodei com faixa etária, escolaridade e renda sendo dummies como teste de robustez em material complementar.
# Gênero: dummy (0 = feminino, 1 = masculino).
# "Prefiro não responder" (renda) vira dado ausente (NaN).
FAIXA_MAP = {
    "( ) 18–24 anos": 1, "( ) 25–34 anos": 2, "( ) 35–44 anos": 3,
    "( ) 45–54 anos": 4, "( ) 55 anos ou mais": 5,
}
ESC_MAP = {
    "Ensino fundamental incompleto": 1,
    "Ensino fundamental completo": 2,
    "Ensino Médio incompleto": 3,
    "Ensino médio completo": 4,
    "Ensino superior incompleto": 5,
    "Ensino superior completo": 6,
    "Pós-graduação completa ou incompleta (Especializações, Mestrado, Doutorado)": 7,
}
RENDA_MAP = {
    "Até R$ 3.000": 1, "De R$ 3.001 a R$ 7.000": 2,
    "De R$ 7.001 a R$ 15.000": 3, "Acima de R$ 15.000": 4,
    "Prefiro não responder": np.nan,
}

df["idade_ord"] = df["faixa_etaria"].map(FAIXA_MAP)
df["genero_dummy"] = df["genero"].map({"Feminino": 0, "Masculino": 1})
df["escolaridade_ord"] = df["escolaridade"].map(ESC_MAP)
df["renda_ord"] = df["renda"].map(RENDA_MAP)

# Conferência: algum valor ficou sem código? (só "Prefiro não responder" é esperado)
for original, codificada in [("faixa_etaria", "idade_ord"), ("genero", "genero_dummy"),
                             ("escolaridade", "escolaridade_ord"), ("renda", "renda_ord")]:
    sem_codigo = df[df[codificada].isna() & (df[original] != "Prefiro não responder")][original]
    if len(sem_codigo) > 0:
        print(f"ATENÇÃO - valores sem código em {original}:", sem_codigo.unique())

print("Dados ausentes por variável:")
print(df[["idade_ord", "genero_dummy", "escolaridade_ord", "renda_ord"]].isna().sum())


# %% 4. TABELA 1 - Perfil da amostra
def tabela_frequencia(serie, ordem, nome):
    """Frequência (N) e percentual de uma variável, na ordem natural das categorias."""
    contagem = serie.value_counts().reindex(ordem)
    out = pd.DataFrame({"Categoria": ordem,
                        "%": (contagem.values / len(serie) * 100).round(2),
                        "N": contagem.values.astype(int)})
    out.insert(0, "Variável", nome)
    return out

# Ordem natural das categorias (do menor para o maior)
ordem_faixa = list(FAIXA_MAP.keys())
ordem_genero = ["Feminino", "Masculino"]
ordem_esc = list(ESC_MAP.keys())
ordem_renda = ["Até R$ 3.000", "De R$ 3.001 a R$ 7.000",
               "De R$ 7.001 a R$ 15.000", "Acima de R$ 15.000", "Prefiro não responder"]

tabela1 = pd.concat([
    tabela_frequencia(df["faixa_etaria"], ordem_faixa, "Faixa etária"),
    tabela_frequencia(df["genero"], ordem_genero, "Gênero"),
    tabela_frequencia(df["escolaridade"], ordem_esc, "Escolaridade"),
    tabela_frequencia(df["renda"], ordem_renda, "Renda familiar mensal"),
], ignore_index=True)
tabela1["Categoria"] = tabela1["Categoria"].str.replace("( ) ", "", regex=False)

tabelas["Tabela 1"] = tabela1
print("\nTABELA 1. Perfil da amostra")
print(tabela1.to_string(index=False))


# %% 5. TABELA 2 - Estatísticas descritivas dos construtos
tabela2 = (df[["IPRS", "ECS", "CC"]]
           .agg(["mean", "std", "min", "max"]).T
           .round(2)
           .rename(columns={"mean": "Média", "std": "Desvio-padrão",
                            "min": "Mínimo", "max": "Máximo"}))
tabela2.index.name = "Construto"
tabelas["Tabela 2"] = tabela2.reset_index()

print("\nTABELA 2. Estatísticas descritivas dos construtos")
print(tabela2)


# PARTE 2 - Confiabilidade e validade dos construtos (Tabelas 3 a 6)

# %% 6. Bibliotecas desta parte
from scipy import stats
from sklearn.decomposition import PCA

#Spicy: traz funções estatísticas. Aqui é usada para a correlação ponto-bisserial (stats.pointbiserialr) na Tabela 6 e, nas próximas partes, para Pearson, Spearman e testes de hipótese.
# sklearn.decomposition: traz a Análise de Componentes Principais do scikit-learn. Foi usada para extrair a primeira componente dos itens de cada construto e, com as cargas, calcular a AVE (Tabela 4).

# %% 7. TABELA 3 - Alfa de Cronbach (conferência do cálculo feito no Excel)
def alfa_cronbach(itens):
    """Alfa de Cronbach = k/(k-1) * (1 - soma das variâncias dos itens / variância da soma)."""
    itens = itens.apply(pd.to_numeric).dropna()
    k = itens.shape[1]
    soma_var_itens = itens.var(ddof=1).sum()
    var_soma = itens.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - soma_var_itens / var_soma)
 
tabela3 = pd.DataFrame({
    "Construto": ["IPRS", "ECS", "CC"],
    "Nº de itens": [len(iprs_items), len(ecs_items), len(cc_items)],
    "Alfa de Cronbach": [alfa_cronbach(df[iprs_items]),
                         alfa_cronbach(df[ecs_items]),
                         alfa_cronbach(df[cc_items])],
}).round(2)
 
tabelas["Tabela 3"] = tabela3
print("TABELA 3. Alfa de Cronbach")      # esperado: 0,90 | 0,81 | 0,88
print(tabela3.to_string(index=False))
 
 
# %% 8. TABELA 4 - Variância Média Extraída (AVE) por PCA
def calcular_ave(itens):
    
    # AVE para ajudar a entender o conceito: 
    # quanto do que as perguntas do construto dizem é "a mesma coisa" (o construto) e
    # quanto é ruído de cada pergunta?
    
    """
    AVE aproximada pela primeira componente principal dos itens padronizados.
    carga = peso do item na componente; AVE = média das cargas ao quadrado.
    (Aproximação: a PCA tende a superestimar a AVE de um modelo de mensuração.)
    """
    itens = itens.apply(pd.to_numeric).dropna()
    z = (itens - itens.mean()) / itens.std()          # padronização (média 0, desvio 1)
    
    # O código acima coloca todas as perguntas na mesma régua.
    # Para cada pergunta, o código subtrai a média e divide pelo desvio padrão.
    # Depois disso, toda pergunta tem média 0 e variação "padrão" (desvio 1). 
    # Sem isso, uma pergunta com respostas muito espalhadas pesaria mais do que uma com respostas concentradas, só por causa da escala.
    
    pca = PCA(n_components=1).fit(z)
    # A PCA procura a única linha que melhor resume as perguntas do construto juntas.
    # Exemplo: É como pegar 5 notas de um aluno e criar uma "nota geral" que represente bem todas.
    # O n_components=1 diz: "quero só essa nota geral". Se as perguntas realmente medem a mesma coisa, ela resume bem. 
    # Se medem coisas diferentes, resume mal.
    
    
    cargas = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    # Calcula a carga de cada pergunta, que diz o quanto cada uma "combina" com essa nota geral. Vai de 0 a 1:
    # perto de 1: a pergunta se alinha muito com o construto; perto de 0: a pergunta quase não tem relação com ele.

    return np.mean(cargas ** 2), cargas
    
    # Eleva cada carga ao quadrado e tira a média. 
    # O quadrado transforma a carga em "fatia da pergunta que é explicada pelo construto".
    #Por exemplo, carga 0,8 vira 0,64, ou seja, 64% do que aquela pergunta mede é o construto e os outros 36% são ruído. A média dessas fatias, entre todas as perguntas, é a AVE.
 
ave = {}
for nome, itens in [("IPRS", iprs_items), ("ECS", ecs_items), ("CC", cc_items)]:
    ave[nome], _ = calcular_ave(df[itens])
 
tabela4 = pd.DataFrame({
    "Construto": list(ave.keys()),
    "AVE": list(ave.values()),
    "√AVE": np.sqrt(list(ave.values())),
}).round(3)
 
tabelas["Tabela 4"] = tabela4
print("\nTABELA 4. Variância Média Extraída (AVE)")   # esperado: 0,708 | 0,364 | 0,541
print(tabela4.to_string(index=False))
 
 
# %% 9. TABELA 5 - Critério de Fornell-Larcker
# Para cada par de construtos, compara-se a correlação com a MENOR √AVE do par
# (teste mais conservador).
corr_construtos = df[["IPRS", "ECS", "CC"]].corr()
raiz_ave = {nome: np.sqrt(valor) for nome, valor in ave.items()}
 
linhas = []
for a, b in [("IPRS", "ECS"), ("IPRS", "CC"), ("ECS", "CC")]:
    menor = a if raiz_ave[a] <= raiz_ave[b] else b          # construto com menor √AVE
    linhas.append({
        "Comparação": f"{a} vs. {b}",
        "√AVE (menor do par)": round(raiz_ave[menor], 3),
        "Construto da √AVE": menor,
        "Correlação": round(corr_construtos.loc[a, b], 3),
        "Margem (√AVE - correlação)": round(raiz_ave[menor] - corr_construtos.loc[a, b], 3),
        "Atende ao critério?": raiz_ave[menor] > corr_construtos.loc[a, b],
    })
 
tabela5 = pd.DataFrame(linhas)
tabelas["Tabela 5"] = tabela5
print("\nTABELA 5. Critério de Fornell-Larcker")
print(tabela5.to_string(index=False))
# esperado: IPRS-ECS 0,604 vs 0,538 | IPRS-CC 0,736 vs 0,731 (margem ~0,005) | ECS-CC 0,604 vs 0,432
 
 
# %% 10. TABELA 6 - Correlação ponto-bisserial com comportamentos de compra declarados

# r é a força e a direção da relação.
# A correlação ponto-bisserial é uma correlação de Pearson em que uma das variáveis é sim/não (1/0) e a outra é uma nota. O r vai de -1 a +1:
# Sinal: positivo significa que quem respondeu "Sim" tende a ter nota mais alta no construto. Negativo significa o contrário.
# Tamanho: quanto mais longe de 0, mais forte a relação. Perto de 0 quase não há relação.
# p é a probabilidade de ver um r assim por acaso. O p não mede o tamanho do efeito.
# r responde "quão forte é a relação?" e o p responde "dá para confiar que ela existe além da amostra?".

def formatar_r_p(r, p):
    """Formata como no trabalho: 0,377 (p<0,001).""" 
    p_txt = "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")
    return f"{r:.3f}".replace(".", ",") + f" ({p_txt})"
 
comportamentos = {
    "Compra influenciada por redes sociais": "compra_influenciada_redes",
    "Compra motivada por boa impressão": "compra_impressionar",
}
 
linhas = []
for rotulo, coluna in comportamentos.items():
    dados = df[[coluna, "IPRS", "ECS", "CC"]].dropna()    #O .dropna() descarta quem não respondeu algum desses campos,
    linha = {"Comportamento declarado": rotulo, "n": len(dados)}
    for construto in ["IPRS", "ECS", "CC"]:
        r, p = stats.pointbiserialr(dados[coluna], dados[construto])
        # Calcula a correlação ponto-bisserial entre o comportamento (que é sim/não, 1/0) e o construto (que é uma nota). A função devolve dois números: r (a força da relação) e p (o p-valor).
        
        linha[construto] = formatar_r_p(r, p)
    linhas.append(linha)
 
tabela6 = pd.DataFrame(linhas)
tabelas["Tabela 6"] = tabela6
print("\nTABELA 6. Correlação ponto-bisserial")
print(tabela6.to_string(index=False))
# esperado: influenciada -> 0,377 (p<0,001) | 0,298 (p=0,004) | 0,280 (p=0,008)
#           boa impressão -> 0,275 (p=0,009) | 0,065 (p=0,543) | 0,508 (p<0,001)
 
# PARTE 3 - Correlações de Pearson e Spearman (Tabelas 7 e 8)

# %% 11. Variáveis da matriz de correlação (mesma ordem das Tabelas 7 e 8)
# Chave = nome que aparece na tabela | valor = coluna do df
variaveis_corr = {
    "IPRS": "IPRS",
    "ECS": "ECS",
    "CC": "CC",
    "Idade": "idade_ord",
    "Gênero": "genero_dummy",
    "Escolaridade": "escolaridade_ord",
    "Renda": "renda_ord",
}
 
# Subconjunto do df só com essas colunas, já com os nomes da tabela
dados_corr = df[list(variaveis_corr.values())].rename(
    columns={coluna: nome for nome, coluna in variaveis_corr.items()}
)
 
 
# %% 12. TABELA 7 - Correlações de Pearson
# .corr() calcula todos os pares de uma vez; cada par usa só as pessoas
# com resposta nas duas variáveis (a renda tem "Prefiro não responder").

#Pearson mede o quanto duas variáveis andam juntas em linha reta,
# usando os valores originais. Por exemplo, se a renda sobe 1 ponto, o CC muda numa quantidade proporcional.

tabela7 = dados_corr.corr(method="pearson").round(3)
 
tabelas["Tabela 7"] = tabela7.reset_index().rename(columns={"index": "Variável"})
print("TABELA 7. Correlações de Pearson")
print(tabela7)
# esperado (conferir com o trabalho):
#   IPRS-ECS 0,538 | IPRS-CC 0,731 | ECS-CC 0,432
#   CC com Idade -0,171 | Gênero 0,074 | Escolaridade 0,083 | Renda -0,129
 
 
# %% 13. TABELA 8 - Correlações de Spearman (checagem de robustez)
# Spearman trabalha com os postos (ranks) dos valores; é mais adequado
# para variáveis ordinais (idade, escolaridade e renda).

# Spearman primeiro troca os valores pela posição (1º, 2º, 3º... do menor para o maior)
# e depois calcula a correlação sobre essas posições. 
# Ele mede se uma variável tende a subir quando a outra sobe, sem exigir que seja em linha reta, só que a ordem se mantenha.


tabela8 = dados_corr.corr(method="spearman").round(3)
 
tabelas["Tabela 8"] = tabela8.reset_index().rename(columns={"index": "Variável"})
print("\nTABELA 8. Correlações de Spearman")
print(tabela8)
# esperado:
#   IPRS-ECS 0,609 | IPRS-CC 0,732 | ECS-CC 0,482
#   CC com Idade -0,184 | Gênero 0,075 | Escolaridade 0,048 | Renda -0,138
 
 
# %% 14. Conferência do texto: diferença entre Spearman e Pearson
# O trabalho afirma "diferenças inferiores a 0,11 em todos os pares".
diferenca = (tabela8 - tabela7).abs()
so_triangulo = np.triu(np.ones(diferenca.shape), k=1).astype(bool)   # evita repetir pares
maiores = diferenca.where(so_triangulo).stack().sort_values(ascending=False)
 
print("\nMaiores diferenças |Spearman - Pearson|:")
print(maiores.head(5).round(3))
print("Maior diferença:", round(maiores.max(), 3))   # esperado: 0,107 (Idade x Escolaridade)
 

#  PARTE 4 - Regressões lineares (Tabelas 9 a 14)

# %% 15. Bibliotecas desta parte
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# É a biblioteca que fornece o sm.OLS para regressões lineares
# statsmodels.stats.outliers_influence é o módulo do statsmodels com ferramentas de diagnóstico do modelo (observações influentes, multicolinearidade).
# Usasdo para cálculo de VIF
 
# %% 16. Funções auxiliares (usadas em todas as regressões)
def ajustar_ols(y_col, x_cols):
    
# ajustar_ols é o motor de todas as regressões

    """
    Ajusta um OLS de y_col sobre x_cols, usando só quem respondeu TODAS as
    variáveis do modelo (dropna). Devolve o modelo e o X (com constante).
    """
    dados = df[[y_col] + x_cols].dropna()
    X = sm.add_constant(dados[x_cols])
    modelo = sm.OLS(dados[y_col], X).fit()   #monta o modelo: y é o CC e X são as variáveis explicativas (com a constante)
    return modelo, X
 
 
def formatar_p(p):
    """p-valor como no trabalho: '<0,001' ou '0,107'."""
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")
 
 
def tabela_coeficientes(modelo, rotulos):
    """Tabela de coeficientes: B, erro-padrão, t e p. 'rotulos' troca os nomes das colunas do df."""
   
# B: quanto o CC muda quando a variável sobe 1 unidade.
# Erro-padrão: a incerteza (imprecisão) do B.
# t: B dividido pelo erro-padrão; mede o tamanho do efeito frente à incerteza.
# p: probabilidade de o resultado ser acaso; abaixo de 0,05 é significativo.
   
    tab = pd.DataFrame({
        "Variável": [rotulos.get(v, v) for v in modelo.params.index],
        "Coeficiente (B)": modelo.params.values,
        "Erro-padrão": modelo.bse.values,
        "t": modelo.tvalues.values,
        "p-valor": [formatar_p(p) for p in modelo.pvalues.values],
    })
    return tab.round({"Coeficiente (B)": 3, "Erro-padrão": 3, "t": 3})
 
 
def estatisticas_modelo(modelo):
    """Estatísticas gerais do modelo, no formato das tabelas do trabalho."""
    return pd.DataFrame({
        "Indicador": ["R²", "R² ajustado", "F", "p(F)", "N"],
        "Valor": [round(modelo.rsquared, 3), round(modelo.rsquared_adj, 4),
                  round(modelo.fvalue, 3), formatar_p(modelo.f_pvalue), int(modelo.nobs)],
    })
 
# R²: a parcela da variação do CC explicada pelo modelo Exemplo (0,535 = 53,5%).
# R² ajustado: o R² corrigido pelo número de variáveis, para não "premiar" modelos com muitas variáveis.
# F: testa se o modelo, como um todo, explica o CC melhor do que nada
# p(F): probabilidade de o resultado do F ser acaso; abaixo de 0,05, o modelo é significativo.
# N: número de respondentes usados no modelo    

# obs: O valor do F em si não tem uma escala fácil de interpretar. O que se usa na prática é o p(F), que diz se ele é grande o bastante para descartar o acaso.

def calcular_vif(X):
    
    vif = pd.DataFrame({
        "Variável": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif[vif["Variável"] != "const"].reset_index(drop=True)
    
 
# calcula o VIF da variável na posição i. (usado na tabela 14)
# Por dentro, ele prevê aquela variável usando todas as outras. Se as outras a "explicam" bem, ela é redundante e o VIF sobe. 
# o quanto a variância do coeficiente desta variável está inflada por causa das outras?" (VIF = Variance Inflation Factor, fator de inflação da variância.) Para cada variável, o cálculo prevê ela mesma a partir das demais:
# VIF = 1 / (1 − R²)   

# Nomes que aparecem nas tabelas
rotulos = {"const": "Constante", "IPRS": "IPRS", "ECS": "ECS",
           "idade_ord": "Idade", "genero_dummy": "Gênero",
           "escolaridade_ord": "Escolaridade", "renda_ord": "Renda"}
 
 
# %% 17. TABELA 9 - Regressão simples: CC ~ IPRS (H1)
m_iprs, _ = ajustar_ols("CC", ["IPRS"])
tabela9 = tabela_coeficientes(m_iprs, rotulos)
est9 = estatisticas_modelo(m_iprs)
 
tabelas["Tabela 9"] = tabela9
tabelas["Tabela 9 - estatísticas"] = est9
print("TABELA 9. CC ~ IPRS")
print(tabela9.to_string(index=False))
print(est9.to_string(index=False))
# esperado: Constante 1,005 (EP 0,152; t 6,595) | IPRS 0,622 (0,062; t 10,054; p<0,001)
#           R² 0,535 | R² aj. 0,5293 | F 101,082 | N 90
 
 
# %% 18. TABELA 10 - Regressão simples: CC ~ ECS (H2)
m_ecs, _ = ajustar_ols("CC", ["ECS"])
tabela10 = tabela_coeficientes(m_ecs, rotulos)
est10 = estatisticas_modelo(m_ecs)
 
tabelas["Tabela 10"] = tabela10
tabelas["Tabela 10 - estatísticas"] = est10
print("\nTABELA 10. CC ~ ECS")
print(tabela10.to_string(index=False))
print(est10.to_string(index=False))
# esperado: Constante 1,259 (0,267; t 4,723) | ECS 0,548 (0,122; t 4,490; p<0,001)
#           R² 0,186 | R² aj. 0,177 | F 20,158 | N 90
 
 
# %% 19. TABELA 11 - Regressões simples, uma para cada variável socioeconômica (H3)
linhas = []
for col in ["idade_ord", "genero_dummy", "escolaridade_ord", "renda_ord"]:
    modelo, _ = ajustar_ols("CC", [col])
    linhas.append({
        "Variável independente": rotulos[col],
        "Coeficiente (B)": modelo.params[col],
        "Erro-padrão": modelo.bse[col],
        "t": modelo.tvalues[col],
        "p-valor": formatar_p(modelo.pvalues[col]),
        "R²": modelo.rsquared,
        "n": int(modelo.nobs),      # renda tem n menor (quem não informou fica de fora)
    })
 
tabela11 = pd.DataFrame(linhas).round({"Coeficiente (B)": 3, "Erro-padrão": 3, "t": 3, "R²": 3})
tabelas["Tabela 11"] = tabela11
print("\nTABELA 11. Regressões simples com variáveis socioeconômicas")
print(tabela11.to_string(index=False))
# esperado: Idade -0,150 (0,092; t -1,629; p 0,107; R² 0,029; n 90)
#           Gênero 0,134 (0,193; 0,694; 0,490; 0,005; 90)
#           Escolaridade 0,062 (0,080; 0,780; 0,438; 0,007; 90)
#           Renda -0,127 (0,108; -1,181; 0,241; 0,017; n 85)
 
 
# %% 20. TABELA 12 - Regressão múltipla: modelo completo (6 variáveis)
vars_completo = ["IPRS", "ECS", "idade_ord", "genero_dummy", "escolaridade_ord", "renda_ord"]
m_completo, X_completo = ajustar_ols("CC", vars_completo)
tabela12 = tabela_coeficientes(m_completo, rotulos)
est12 = estatisticas_modelo(m_completo)
 
tabelas["Tabela 12"] = tabela12
tabelas["Tabela 12 - estatísticas"] = est12
print("\nTABELA 12. Modelo completo (n = %d)" % m_completo.nobs)
print(tabela12.to_string(index=False))
print(est12.to_string(index=False))
# esperado: Constante 0,616 (p 0,293) | IPRS 0,665 (EP 0,076; t 8,758; p<0,001)
#           ECS 0,005 (0,115; 0,044; 0,965) | Idade 0,046 (0,072; 0,639; 0,525)
#           Gênero 0,366 (0,139; 2,627; 0,010) | Escolaridade 0,044 (0,070; 0,622; 0,536)
#           Renda -0,100 (0,082; -1,223; 0,225)
#           R² 0,589 | R² aj. 0,5574 | F 18,632 | N 85
 
 
# %% 21. TABELA 13 - Regressão múltipla: modelo final exploratório (IPRS, ECS e gênero)
vars_final = ["IPRS", "ECS", "genero_dummy"]
m_final, X_final = ajustar_ols("CC", vars_final)
tabela13 = tabela_coeficientes(m_final, rotulos)
est13 = estatisticas_modelo(m_final)
 
tabelas["Tabela 13"] = tabela13
tabelas["Tabela 13 - estatísticas"] = est13
print("\nTABELA 13. Modelo final exploratório (n = %d)" % m_final.nobs)
print(tabela13.to_string(index=False))
print(est13.to_string(index=False))
# esperado: Constante 0,719 (0,207; t 3,479; p 0,001) | IPRS 0,647 (0,072; 8,999; <0,001)
#           ECS 0,037 (0,105; 0,355; 0,724) | Gênero 0,409 (0,130; 3,155; 0,002)
#           R² 0,585 (0,5848) | R² aj. 0,5703 | F 40,368 | N 90
 
# Equação do modelo final (texto do trabalho)
b = m_final.params
print(f"\nCC = {b['const']:.3f} + {b['IPRS']:.3f}(IPRS) + {b['ECS']:.3f}(ECS) + {b['genero_dummy']:.3f}(Gênero)")
 
 
# %% 22. TABELA 14 - VIF (modelo final) e VIF do modelo completo
tabela14 = calcular_vif(X_final).round(2)
tabela14["Variável"] = tabela14["Variável"].map(rotulos)
tabelas["Tabela 14"] = tabela14
print("\nTABELA 14. VIF (modelo final)")        # esperado: IPRS 1,48 | ECS 1,42 | Gênero 1,05
print(tabela14.to_string(index=False))
 
vif_completo = calcular_vif(X_completo)["VIF"]
print("\nVIF do modelo completo: de %.2f a %.2f" % (vif_completo.min(), vif_completo.max()))
# esperado: de 1,09 a 1,53
 
 
# %% 23. Critério de Green (1991) para o tamanho da amostra
# Avaliação do modelo como um todo: N >= 50 + 8m | de cada coeficiente: N >= 104 + m
# (m = número de variáveis explicativas)
for nome, modelo, m in [("Modelo completo", m_completo, len(vars_completo)),
                        ("Modelo final", m_final, len(vars_final))]:
    n = int(modelo.nobs)
    print(f"\n{nome}: n = {n}, m = {m}")
    print(f"  Modelo como um todo (50 + 8m = {50 + 8*m}): atende? {n >= 50 + 8*m}")
    print(f"  Coeficientes (104 + m = {104 + m}): atende? {n >= 104 + m}")
# esperado: completo -> exige 98 e 110, n = 85 (não atende)
#           final    -> exige 74 e 107, n = 90 (atende o primeiro, não o segundo)
 
# PARTE 5 - Exportação das tabelas para o Excel

# %% 24. Exportar as tabelas para o Excel
# A base de dados NÃO é exportada (privacidade dos respondentes).
ARQUIVO_SAIDA = "resultados_consumo_conspicuo.xlsx"

with pd.ExcelWriter(ARQUIVO_SAIDA, engine="openpyxl") as escritor:
    linha = 0
    for nome, tabela in tabelas.items():
        # título da tabela (ex.: "Tabela 7")
        pd.DataFrame([[nome]]).to_excel(escritor, sheet_name="Tabelas",
                                        startrow=linha, index=False, header=False)
        # a própria tabela, uma linha abaixo do título
        tabela.to_excel(escritor, sheet_name="Tabelas", startrow=linha + 1, index=False)
        linha += len(tabela) + 4            # deixa 2 linhas em branco antes da próxima

print(f"Arquivo salvo: {ARQUIVO_SAIDA}")
