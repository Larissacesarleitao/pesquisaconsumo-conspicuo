"""
Material complementar - Verificação com variáveis indicadoras (dummies)
-----------------------------------------------------------------------

No trabalho, faixa etária, escolaridade e renda entram nas regressões como
variáveis ordinais (1, 2, 3...), o que supõe espaçamento igual entre as
categorias. Este código verifica essa suposição reestimando os modelos com
dummies (uma variável 0/1 para cada categoria), sem supor essa igualdade.

  Parte A  Regressões simples de CC com dummies, uma variável por vez
  Parte B  Modelo final (IPRS, ECS e gênero) com as dummies acrescentadas
           uma variável por vez
  Parte C  Média de CC por faixa etária (apoia a leitura do resultado de idade)

"""

# %% 0. Bibliotecas e configurações
import numpy as np
import pandas as pd
import statsmodels.api as sm

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

CAMINHO_XLSX = "Pesquisa - Consumo Conspícuo.xlsx"
ABA = "Base Resultado Final"

tabelas = {}    # tabelas deste material, exportadas no final


# %% 1. Carregar, limpar e codificar (mesmos passos do código principal)
df = pd.read_excel(CAMINHO_XLSX, sheet_name=ABA, header=0)
df = df[df.iloc[:, 0].notna()].copy()        # remove linhas em branco
df = df[df.iloc[:, 1] == "Sim"].copy()       # só maiores de 18 anos
print("N final:", len(df))                   # esperado: 90

cols = df.columns.tolist()
df = df.rename(columns={
    cols[2]: "faixa_etaria", cols[3]: "genero", cols[4]: "escolaridade", cols[5]: "renda",
    cols[20]: "ECS", cols[27]: "IPRS", cols[38]: "CC",
})
for nome in ["ECS", "IPRS", "CC"]:
    df[nome] = df[nome].astype(float)

FAIXA_MAP = {"( ) 18–24 anos": 1, "( ) 25–34 anos": 2, "( ) 35–44 anos": 3,
             "( ) 45–54 anos": 4, "( ) 55 anos ou mais": 5}
ESC_MAP = {"Ensino fundamental incompleto": 1, "Ensino fundamental completo": 2,
           "Ensino Médio incompleto": 3, "Ensino médio completo": 4,
           "Ensino superior incompleto": 5, "Ensino superior completo": 6,
           "Pós-graduação completa ou incompleta (Especializações, Mestrado, Doutorado)": 7}
RENDA_MAP = {"Até R$ 3.000": 1, "De R$ 3.001 a R$ 7.000": 2,
             "De R$ 7.001 a R$ 15.000": 3, "Acima de R$ 15.000": 4,
             "Prefiro não responder": np.nan}

df["idade_ord"] = df["faixa_etaria"].map(FAIXA_MAP)
df["genero_dummy"] = df["genero"].map({"Feminino": 0, "Masculino": 1})
df["escolaridade_ord"] = df["escolaridade"].map(ESC_MAP)
df["renda_ord"] = df["renda"].map(RENDA_MAP)


# %% 2. Funções auxiliares
def fmt(x, casas=3):
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")

def fmt_p(p):
    return "<0,001" if p < 0.001 else fmt(p, 3)

def criar_dummies(serie):
    """
    Transforma uma variável ordinal em dummies (0/1), uma por categoria.
    drop_first=True descarta a 1ª categoria, que vira a categoria de referência
    (sem isso, as dummies somariam sempre 1 e o modelo não se resolveria).
    """
    return pd.get_dummies(serie.astype(int).astype("category"),
                          prefix=serie.name, drop_first=True, dtype=int)

def coef_p(modelo, variavel):
    """Coeficiente (B) e p-valor numa única célula: '0,634; <0,001'."""
    return f"{fmt(modelo.params[variavel])}; {fmt_p(modelo.pvalues[variavel])}"

# Variáveis ordinais que serão transformadas em dummies
variaveis_ordinais = {"Faixa etária": "idade_ord",
                      "Escolaridade": "escolaridade_ord",
                      "Renda": "renda_ord"}


# %% 3. PARTE A - Regressões simples com dummies (uma variável por vez)
# O teste F do modelo avalia se o CONJUNTO de dummies explica CC.
linhas = []
for nome, col in variaveis_ordinais.items():
    d = df[["CC", col]].dropna()          # retira quem não respondeu (ex.: renda)
    modelo = sm.OLS(d["CC"], sm.add_constant(criar_dummies(d[col]))).fit()
    linhas.append({
        "Variável": nome,
        "Categorias": d[col].nunique(),
        "n": int(modelo.nobs),
        "gl": int(modelo.df_model),       # nº de dummies
        "R²": fmt(modelo.rsquared),
        "R² ajustado": fmt(modelo.rsquared_adj),
        "F": fmt(modelo.fvalue, 2),
        "p": fmt_p(modelo.f_pvalue),
    })

tabela_A = pd.DataFrame(linhas)
tabelas["Tabela A1 - Dummies isoladas"] = tabela_A
print("TABELA A1. Regressões simples de CC com dummies (teste F do conjunto)")
print(tabela_A.to_string(index=False))
# esperado (texto do trabalho):
#   Faixa etária  F = 2,67; p = 0,037
#   Escolaridade  F = 1,13; p = 0,353
#   Renda         F = 2,45; p = 0,069


# %% 4. PARTE B - Modelo final + dummies de cada variável (uma por vez)
base_final = ["IPRS", "ECS", "genero_dummy"]       # modelo final do trabalho

# Linha de referência: modelo final sem dummies
d0 = df[["CC"] + base_final].dropna()
m0 = sm.OLS(d0["CC"], sm.add_constant(d0[base_final])).fit()
linhas = [{
    "Dummies acrescentadas": "Nenhuma (modelo final)", "n": int(m0.nobs),
    "gl": "—", "F": "—", "p (conjunto)": "—",
    "IPRS (B; p)": coef_p(m0, "IPRS"),
    "ECS (B; p)": coef_p(m0, "ECS"),
    "Gênero (B; p)": coef_p(m0, "genero_dummy"),
}]

for nome, col in variaveis_ordinais.items():
    d = df[["CC"] + base_final + [col]].dropna()   # mesma amostra nos dois modelos
    m_sem = sm.OLS(d["CC"], sm.add_constant(d[base_final])).fit()                 # sem as dummies
    X = sm.add_constant(pd.concat([d[base_final], criar_dummies(d[col])], axis=1))
    m_com = sm.OLS(d["CC"], X).fit()                                              # com as dummies

    # Teste F comparando os dois modelos: as dummies, em conjunto, acrescentam algo?
    F, p, gl = m_com.compare_f_test(m_sem)
    linhas.append({
        "Dummies acrescentadas": nome, "n": int(m_com.nobs), "gl": int(gl),
        "F": fmt(F, 2), "p (conjunto)": fmt_p(p),
        "IPRS (B; p)": coef_p(m_com, "IPRS"),
        "ECS (B; p)": coef_p(m_com, "ECS"),
        "Gênero (B; p)": coef_p(m_com, "genero_dummy"),
    })

tabela_B = pd.DataFrame(linhas)
tabelas["Tabela A2 - Dummies no modelo final"] = tabela_B
print("\nTABELA A2. Modelo final com as dummies acrescentadas, uma de cada vez")
print(tabela_B.to_string(index=False))
# esperado (texto do trabalho):
#   Idade F = 1,64 (p = 0,172) | Escolaridade F = 0,88 (p = 0,512) | Renda F = 1,75 (p = 0,164)
#   IPRS: B de 0,634 a 0,653 (p < 0,001) | Gênero: B de 0,376 a 0,423 (p <= 0,007)
#   ECS: B de -0,038 a 0,027 (p >= 0,732)


# %% 5. PARTE C - Média de CC por faixa etária
# Mostra de onde vem o padrão não linear de idade (menor CC em 45-54 anos)
# e que essa faixa tem poucos respondentes (média pouco precisa).
por_faixa = (df.groupby("faixa_etaria")["CC"]
               .agg(n="count", media="mean", desvio_padrao="std")
               .reindex(list(FAIXA_MAP.keys()))
               .round(3)
               .reset_index())
tabelas["Tabela A3 - CC por faixa etária"] = por_faixa
print("\nTABELA A3. Consumo Conspícuo (CC) por faixa etária")
print(por_faixa.to_string(index=False))
# esperado: a faixa de 45 a 54 anos tem n = 7 e a menor média


# %% 6. Exportar para o Excel
ARQUIVO_SAIDA = "material_complementar_dummies.xlsx"

with pd.ExcelWriter(ARQUIVO_SAIDA, engine="openpyxl") as escritor:
    linha = 0
    for nome, tabela in tabelas.items():
        pd.DataFrame([[nome]]).to_excel(escritor, sheet_name="Tabelas",
                                        startrow=linha, index=False, header=False)
        tabela.to_excel(escritor, sheet_name="Tabelas", startrow=linha + 1, index=False)
        linha += len(tabela) + 4

print(f"\nArquivo salvo: {ARQUIVO_SAIDA}")
