"""
Diagnóstico dos modelos de regressão do TCC
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from scipy import stats       #funções científicas e estatísticas.

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)

CAMINHO = "Pesquisa - Consumo Conspícuo.xlsx"
SAIDA = "."       #"." = pasta atual

df = pd.read_excel(CAMINHO, sheet_name="Base Resultado Final", header=0)   #primeira linha da planilha tem os nomes das colunas

#tratativas da base
df = df[df.iloc[:, 0].notna()].copy() #Verifia carimbo data e hora quem preencheu, cria uma cópia com o filtro.
df = df[df.iloc[:, 1] == "Sim"].copy() #Maiores de 18 anos 
cols = df.columns.tolist()


#organização das variáveis: Troca o nome longo de cada coluna por um nome curto
df = df.rename(columns={cols[2]: "faixa_etaria", cols[3]: "genero", cols[4]: "escolaridade",
                        cols[5]: "renda", cols[20]: "ECS", cols[27]: "IPRS", cols[38]: "CC"})

#Preparo para a regressão
for c in ["ECS", "IPRS", "CC"]: df[c] = df[c].astype(float)     
FAIXA = {"( ) 18–24 anos": 1, "( ) 25–34 anos": 2, "( ) 35–44 anos": 3, "( ) 45–54 anos": 4, "( ) 55 anos ou mais": 5}
ESC = {"Ensino fundamental incompleto": 1, "Ensino fundamental completo": 2, "Ensino Médio incompleto": 3,
       "Ensino médio completo": 4, "Ensino superior incompleto": 5, "Ensino superior completo": 6,
       "Pós-graduação completa ou incompleta (Especializações, Mestrado, Doutorado)": 7}
RENDA = {"Até R$ 3.000": 1, "De R$ 3.001 a R$ 7.000": 2, "De R$ 7.001 a R$ 15.000": 3,
         "Acima de R$ 15.000": 4, "Prefiro não responder": np.nan}
df["idade_ord"] = df["faixa_etaria"].map(FAIXA)
df["genero_dummy"] = df["genero"].map({"Feminino": 0, "Masculino": 1})
df["escolaridade_ord"] = df["escolaridade"].map(ESC)
df["renda_ord"] = df["renda"].map(RENDA)
df = df.reset_index(drop=True)
print("N =", len(df))




MODELOS = {
    "FINAL (IPRS, ECS, genero)": ["IPRS", "ECS", "genero_dummy"],
    "COMPLETO (6 variaveis)": ["IPRS", "ECS", "idade_ord", "genero_dummy", "escolaridade_ord", "renda_ord"],
    "SIMPLES IPRS": ["IPRS"],
    "SIMPLES ECS": ["ECS"],
}



for nome, variaveis in MODELOS.items():    # variaveis =  lista de variáveis explicativas do modelo da vez
    d = df[["CC"] + variaveis].dropna()
    X = sm.add_constant(d[variaveis]);
    m = sm.OLS(d["CC"], X).fit()   #modelo ajustado
    n, k = int(m.nobs), len(variaveis)   #n = número de respondentes usados no modelo e K=numero de variáveis do modelo


# formato para dados do modelo de regressão
    print()
    print("MODELO:", nome)
    print("n =", n)
    print("R² =", round(m.rsquared, 4))
    print("R² ajustado =", round(m.rsquared_adj, 4))
    print("F =", round(m.fvalue, 3))    

# heterocedasticidade
    bp = het_breuschpagan(m.resid, X); wh = het_white(m.resid, X) if k > 1 else (np.nan,)*4   #het_white = roda o teste de White só quando o modelo tem mais de uma variável.
# Obs.: o White mostra o aviso "SingularMatrixWarning" nos modelos com gênero. É esperado e não afeta o resultado:
# o teste usa o quadrado de cada variável, e o quadrado da dummy de gênero (0/1) é igual a ela mesma.
# Normalidade
    sw = stats.shapiro(m.resid)
# linearidade
    reset2 = linear_reset(m, power=2, use_f=True)  #power=2: acrescenta os valores previstos ao quadrado (reportado no TCC).
# Observações influentes
    infl = m.get_influence(); cook = infl.cooks_distance[0]   #distância de Cook de cada respondente


# Resultados dos testes 

    print(f"Breusch-Pagan  LM = {bp[0]:.3f}, p = {bp[1]:.3f}")
    if k > 1: print(f"White          LM = {wh[0]:.3f}, p = {wh[1]:.3f}")

    print(f"Shapiro-Wilk   W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}")
     
    print(f"Assimetria = {stats.skew(m.resid):.2f} | Curtose (excesso) = {stats.kurtosis(m.resid):.2f}")

    print(f"RESET  F = {float(reset2.fvalue):.3f}, p = {float(reset2.pvalue):.3f}")

    print(f"Cook: max = {cook.max():.3f} | > 4/n ({4/n:.3f}): {(cook > 4/n).sum()} obs | > 1: {(cook > 1).sum()}")


# Erros-padrão robustos
    rob = m.get_robustcov_results("HC3")
    comp = pd.DataFrame({"B": m.params, "EP OLS": m.bse, "p OLS": m.pvalues,
                         "EP HC3": rob.bse, "p HC3": rob.pvalues,
                         "IC95 inf": m.conf_int()[0], "IC95 sup": m.conf_int()[1]})
   
    
    print(comp.round(4))


# Observações influentes: quem tem distância de Cook acima de 4/n
    influentes = cook > 4/n                  #marca True para quem passa do limite
    qtd_influentes = influentes.sum()        #conta quantos são

# Se houver influentes, roda o mesmo modelo sem eles para ver se o resultado muda
    if qtd_influentes > 0:
        d_sem = d[~influentes]               #base sem os influentes (~ significa "não")
        X_sem = sm.add_constant(d_sem[variaveis])
        m_sem = sm.OLS(d_sem["CC"], X_sem).fit()

        print()
        print("Sem as", qtd_influentes, "observações influentes:")
        print("n =", int(m_sem.nobs))
        print("R² =", round(m_sem.rsquared, 3))
        print(pd.DataFrame({"B": m_sem.params, "p": m_sem.pvalues}).round(4))

        # Lista de quem são os influentes, do maior Cook para o menor
        lista = d[influentes].copy()
        lista["cook"] = cook[influentes]
        lista["previsto"] = m.fittedvalues[influentes]
        print()
        print("Observações influentes:")
        print(lista.sort_values("cook", ascending=False).round(3))


# Figura com os 3 gráficos de diagnóstico (só para o modelo final)
    if "FINAL" in nome:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))     #1 linha com 3 gráficos

        # Gráfico 1: resíduos x valores previstos (heterocedasticidade e linearidade)
        ax[0].scatter(m.fittedvalues, m.resid)
        ax[0].axhline(0, color="black")
        ax[0].set_title("Resíduos x valores previstos")
        ax[0].set_xlabel("Valores previstos")
        ax[0].set_ylabel("Resíduos")

        # Gráfico 2: Q-Q dos resíduos (normalidade)
        sm.qqplot(m.resid, line="s", ax=ax[1])
        ax[1].set_title("Q-Q dos resíduos")
        ax[1].set_xlabel("Quantis teóricos")
        ax[1].set_ylabel("Quantis observados")

        # Gráfico 3: distância de Cook de cada respondente (observações influentes)
        ax[2].vlines(range(n), 0, cook)
        ax[2].axhline(4/n, color="red", linestyle="--", label="4/n")
        ax[2].set_title("Observações influentes")
        ax[2].set_xlabel("Observação")
        ax[2].set_ylabel("Distância de Cook")
        ax[2].legend()

        plt.tight_layout()
        plt.savefig(f"{SAIDA}/diagnostico_modelo.png", dpi=200)

     
