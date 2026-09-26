"""
A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
Evidências dos Ciclos da Selic (2005-2024)

M&A_t = α + β1 M&A_(t-1) + β2 Selic_(t-k) + β3 RetornoIbov_t + β4 Covid_t + ε_t

Modelo MQO conforme o plano de trabalho, 3 defasagens da Selic (1, 3, 6 meses).
Saídas em ./resultados.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # salva figuras em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO  (todo critério de amostra/modelo fica aqui, explícito)
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados"
SAIDA.mkdir(exist_ok=True)

# Critérios de amostra (aplicados sobre as colunas NUMÉRICAS brutas, nunca
# sobre as colunas "Filtro .../Sim" da planilha — ver AUDITORIA no histórico
# do projeto: usar a coluna de % como máscara de texto infla a amostra de
# 2.292 para 6.571 deals).
DEAL_MIN_USD_MM = 0.5      # Deal Size (USD, MM) > 0,5
PCT_MIN = 50.0             # % of Target Shares Sought > 50 (participação de controle)
REMOVER_DUPLICATAS = False  # 11 linhas suspeitas; decisão em aberto com o aluno

# Janela da amostra
INICIO, FIM = "2005-01", "2024-12"

# Defasagens da Selic a testar (um modelo para cada, mesma amostra e controles)
LAGS_SELIC = [1, 3, 6]

# Dummy da Covid-19: recorte do choque inicial (mar-dez/2020), como no plano
# de trabalho. Testado também com uma janela alternativa na robustez.
COVID_INICIO, COVID_FIM = "2020-03", "2020-12"

NW_LAGS = 6  # defasagens do erro-padrão HAC (Newey-West), para corrigir
             # autocorrelação/heterocedasticidade residual em séries mensais

# =============================================================================
# 1. DADOS: leitura, filtros de amostra e painel mensal
# =============================================================================


def carregar_negocios() -> pd.DataFrame:
    """Lê a base bruta FactSet e prepara as colunas numéricas de filtro."""
    df = pd.read_excel(ARQUIVO, sheet_name="Transactions_Results", header=2)
    df["Announcement Date"] = pd.to_datetime(df["Announcement Date"])
    df["valor"] = pd.to_numeric(df["Deal Size (USD, MM)"], errors="coerce")
    df["pct"] = pd.to_numeric(df["% of Target Shares Sought"], errors="coerce")
    return df


def aplicar_filtros(df: pd.DataFrame, deal_min=DEAL_MIN_USD_MM, pct_min=PCT_MIN,
                    dedup=REMOVER_DUPLICATAS) -> pd.DataFrame:
    m = (df["Deal Status"] == "Completed") & (df["Country (Target/Issuer)"] == "Brazil")
    if deal_min is not None:
        m &= df["valor"] > deal_min
    if pct_min is not None:
        m &= df["pct"] > pct_min
    out = df[m].copy()
    if dedup:
        out = out.drop_duplicates(
            ["Company Name (Target/Issuer)", "Announcement Date", "valor"])
    return out


def contagem_mensal(negocios: pd.DataFrame) -> pd.Series:
    """Nº de anúncios por mês (meses sem operação = 0)."""
    per = negocios["Announcement Date"].dt.to_period("M")
    s = per.value_counts().sort_index()
    idx = pd.period_range(INICIO, FIM, freq="M")
    return s.reindex(idx, fill_value=0).rename("N_MA")


def _serie_macro(m: pd.DataFrame, col_data: int, col_val: int, nome: str) -> pd.Series:
    s = m.iloc[:, [col_data, col_val]].copy()
    s.columns = ["data", nome]
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    s = s.dropna()
    s.index = s["data"].dt.to_period("M")
    return pd.to_numeric(s[nome], errors="coerce")


def carregar_macro() -> pd.DataFrame:
    """Séries mensais da aba Macro_Controls (posições das colunas na planilha)."""
    m = pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)
    macro = pd.concat([
        _serie_macro(m, 0, 1, "Selic"),       # Selic BCB, % a.m. acumulada no mês
        _serie_macro(m, 7, 8, "IPCA"),
        _serie_macro(m, 13, 14, "IBC_Br"),
        _serie_macro(m, 19, 20, "IIE_Br"),
        _serie_macro(m, 28, 29, "Ibov_Nivel"),
        _serie_macro(m, 28, 30, "Ibov_Vol"),
    ], axis=1)
    macro["Ibov_Ret"] = 100 * np.log(macro["Ibov_Nivel"]).diff()  # retorno mensal (%)
    return macro


def montar_painel(negocios: pd.DataFrame) -> pd.DataFrame:
    painel = pd.concat([contagem_mensal(negocios), carregar_macro()], axis=1)
    painel = painel.loc[INICIO:FIM]

    # Termo autorregressivo (Grönroos usa Ln(Number) contemporâneo; aqui
    # entra M&A_(t-1) explicitamente, como no plano de trabalho)
    painel["N_MA_lag1"] = painel["N_MA"].shift(1)

    # Dummy Covid-19 (choque inicial, mar-dez/2020)
    idx_str = painel.index.astype(str)
    painel["Covid"] = ((idx_str >= COVID_INICIO) & (idx_str <= COVID_FIM)).astype(int)

    for k in LAGS_SELIC:
        painel[f"Selic_lag{k}"] = painel["Selic"].shift(k)

    obrigatorias = ["N_MA", "N_MA_lag1", "Ibov_Ret", "Covid"] + \
        [f"Selic_lag{k}" for k in LAGS_SELIC]
    faltando = painel[obrigatorias].isna().sum()
    painel.index.name = "Periodo"
    return painel, faltando


# =============================================================================
# 2. ESTATÍSTICA DESCRITIVA E ESTACIONARIEDADE
# =============================================================================


def estatisticas_descritivas(painel: pd.DataFrame) -> pd.DataFrame:
    cols = ["N_MA", "Selic", "Ibov_Ret", "IPCA", "IBC_Br", "IIE_Br"]
    return painel[cols].describe().T


def grafico_series(painel: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(painel.index.to_timestamp(), painel["N_MA"], color="#1f4e79")
    ax[0].set_title("Número mensal de anúncios de M&A")
    ax[0].axvspan(pd.Period(COVID_INICIO).to_timestamp(), pd.Period(COVID_FIM).to_timestamp(),
                  color="gray", alpha=.2, label="Covid-19 (recorte)")
    ax[0].legend(loc="upper left")
    ax[1].plot(painel.index.to_timestamp(), painel["Selic"], color="#b7472a")
    ax[1].set_title("Selic mensal acumulada (% a.m.)")
    fig.tight_layout()
    fig.savefig(SAIDA / "series_temporais.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def testes_raiz_unitaria(painel: pd.DataFrame) -> pd.DataFrame:
    """ADF em nível para as variáveis do modelo (Passo 2 do plano de trabalho)."""
    linhas = []
    for v in ["N_MA", "Selic", "Ibov_Ret"]:
        serie = painel[v].dropna()
        adf = adfuller(serie, regression="c", autolag="AIC")
        linhas.append({
            "Variável": v, "ADF stat": adf[0], "ADF p-valor": adf[1],
            "Nº de defasagens (AIC)": adf[2],
            "Estacionária a 5%": adf[1] < .05,
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 3. MODELOS MQO (uma regressão por defasagem da Selic)
# =============================================================================


def amostra_comum(painel: pd.DataFrame) -> pd.DataFrame:
    """Mesma amostra de meses nos três modelos (plano de trabalho, item 4):
    remove as linhas iniciais perdidas pela maior defasagem usada (Selic_lag6
    e N_MA_lag1), para que os três modelos sejam estritamente comparáveis."""
    maior_lag = max(LAGS_SELIC)
    cols = ["N_MA", "N_MA_lag1", "Ibov_Ret", "Covid"] + [f"Selic_lag{k}" for k in LAGS_SELIC]
    return painel[cols].iloc[maior_lag:].copy()


def estimar_modelo(dados: pd.DataFrame, lag_selic: int, nw_lags: int = NW_LAGS):
    y = dados["N_MA"]
    X = dados[["N_MA_lag1", f"Selic_lag{lag_selic}", "Ibov_Ret", "Covid"]].rename(
        columns={f"Selic_lag{lag_selic}": "Selic_lag"})
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit()
    # Erros-padrão HAC (Newey-West): corrige autocorrelação/heterocedasticidade
    # residual típica de séries mensais, sem alterar os coeficientes pontuais.
    # get_robustcov_results devolve arrays "crus" (sem nomes); recolocamos os
    # nomes das colunas de X para poder indexar por nome de variável.
    bruto_hac = ols.get_robustcov_results(cov_type="HAC", maxlags=nw_lags)
    params = pd.Series(bruto_hac.params, index=X.columns)
    bse = pd.Series(bruto_hac.bse, index=X.columns)
    ci = pd.DataFrame(bruto_hac.conf_int(alpha=0.05), index=X.columns, columns=["ci_low", "ci_high"])
    robusto = type("HAC", (), {
        "params": params,
        "pvalues": pd.Series(bruto_hac.pvalues, index=X.columns),
        "bse": bse,
        "conf_int": ci,
        "summary": bruto_hac.summary,
    })()
    return ols, robusto, X, y


def diagnosticos_modelo(ols, X, y) -> dict:
    resid = ols.resid
    bp = het_breuschpagan(resid, ols.model.exog)
    jb = jarque_bera(resid)
    dw = durbin_watson(resid)
    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns)
    return {
        "R2": ols.rsquared, "R2_ajustado": ols.rsquared_adj,
        "F p-valor": ols.f_pvalue, "n": int(ols.nobs),
        "Breusch-Pagan p-valor": bp[1],
        "Jarque-Bera p-valor": jb[1],
        "Durbin-Watson": dw,
        "VIF máximo (exceto const.)": vif.drop("const").max(),
    }


def tabela_resultados(modelos: dict) -> pd.DataFrame:
    """Uma coluna por modelo (defasagem da Selic), coeficientes com erro-padrão
    HAC (Newey-West), p-valor e IC 95% — no formato de tabela de resultados de
    MQO (o IC do coeficiente da Selic atende ao item 4 do plano de trabalho:
    comparar sinal, magnitude e intervalo de confiança entre os 3 modelos)."""
    linhas = {}
    for nome, (ols, robusto, X, y) in modelos.items():
        s, p, ci = robusto.params, robusto.pvalues, robusto.conf_int
        col = {}
        for var in X.columns:
            estrela = "***" if p[var] < .01 else "**" if p[var] < .05 else "*" if p[var] < .10 else ""
            col[var] = f"{s[var]:.4f}{estrela} ({p[var]:.3f}) [{ci.loc[var,'ci_low']:.4f}; {ci.loc[var,'ci_high']:.4f}]"
        d = diagnosticos_modelo(ols, X, y)
        col.update({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in d.items()})
        linhas[nome] = col
    return pd.DataFrame(linhas)


def tabela_comparacao_selic(modelos: dict) -> pd.DataFrame:
    """Item 4 do plano de trabalho: comparar sinal, magnitude e IC 95% do
    coeficiente da Selic entre os 3 modelos, lado a lado, numa tabela dedicada
    (mais fácil de ler do que a tabela_resultados completa)."""
    linhas = []
    for nome, (ols, robusto, X, y) in modelos.items():
        var = "Selic_lag"
        linhas.append({
            "Modelo": nome,
            "Coeficiente Selic": robusto.params[var],
            "Erro-padrão (HAC)": robusto.bse[var],
            "p-valor (HAC)": robusto.pvalues[var],
            "IC 95% inferior": robusto.conf_int.loc[var, "ci_low"],
            "IC 95% superior": robusto.conf_int.loc[var, "ci_high"],
            "Significativo a 5%": bool(robusto.pvalues[var] < .05),
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 4. ROBUSTEZ
# =============================================================================


def robustez(painel: pd.DataFrame, dados: pd.DataFrame) -> pd.DataFrame:
    linhas = []

    # (a) Janela alternativa da dummy Covid: só o choque agudo (mar-jun/2020)
    idx_str = painel.index.astype(str)
    covid_curta = pd.Series(
        ((idx_str >= "2020-03") & (idx_str <= "2020-06")).astype(int), index=painel.index)
    covid_longa = pd.Series(
        ((idx_str >= "2020-03") & (idx_str <= "2021-12")).astype(int), index=painel.index)
    maior_lag = max(LAGS_SELIC)
    for nome, serie in [("Covid curta (mar-jun/2020)", covid_curta),
                        ("Covid longa (mar/2020-dez/2021)", covid_longa)]:
        d = dados.copy()
        d["Covid"] = serie.iloc[maior_lag:].values
        for k in LAGS_SELIC:
            ols, robusto, X, y = estimar_modelo(d, k)
            linhas.append({"Especificação": f"{nome} | Selic_lag{k}",
                           "coef Selic (HAC)": robusto.params["Selic_lag"],
                           "p-valor (HAC)": robusto.pvalues["Selic_lag"],
                           "R2 ajustado": ols.rsquared_adj, "n": int(ols.nobs)})

    # (b) Controle adicional: IPCA contemporâneo (Gonzalez 2024: juros e inflação)
    for k in LAGS_SELIC:
        d = dados.copy()
        d["IPCA"] = painel["IPCA"].iloc[maior_lag:].values
        y = d["N_MA"]
        X = sm.add_constant(d[["N_MA_lag1", f"Selic_lag{k}", "Ibov_Ret", "Covid", "IPCA"]])
        m = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HAC", maxlags=NW_LAGS)
        params = pd.Series(m.params, index=X.columns)
        pvalues = pd.Series(m.pvalues, index=X.columns)
        linhas.append({"Especificação": f"+ IPCA contemporâneo | Selic_lag{k}",
                       "coef Selic (HAC)": params[f"Selic_lag{k}"],
                       "p-valor (HAC)": pvalues[f"Selic_lag{k}"],
                       "R2 ajustado": np.nan, "n": int(m.nobs)})

    # (c) Controle adicional: IIE-Br contemporâneo (Batista et al. 2022: incerteza)
    for k in LAGS_SELIC:
        d = dados.copy()
        d["IIE_Br"] = painel["IIE_Br"].iloc[maior_lag:].values
        y = d["N_MA"]
        X = sm.add_constant(d[["N_MA_lag1", f"Selic_lag{k}", "Ibov_Ret", "Covid", "IIE_Br"]])
        m = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HAC", maxlags=NW_LAGS)
        params = pd.Series(m.params, index=X.columns)
        pvalues = pd.Series(m.pvalues, index=X.columns)
        linhas.append({"Especificação": f"+ IIE-Br contemporâneo | Selic_lag{k}",
                       "coef Selic (HAC)": params[f"Selic_lag{k}"],
                       "p-valor (HAC)": pvalues[f"Selic_lag{k}"],
                       "R2 ajustado": np.nan, "n": int(m.nobs)})

    # (d) Amostra sem duplicatas suspeitas (11 linhas)
    bruto = carregar_negocios()
    neg_sem_dup = aplicar_filtros(bruto, dedup=True)
    p2, _ = montar_painel(neg_sem_dup)
    d2 = amostra_comum(p2)
    for k in LAGS_SELIC:
        ols, robusto, X, y = estimar_modelo(d2, k)
        linhas.append({"Especificação": f"Sem duplicatas | Selic_lag{k}",
                       "coef Selic (HAC)": robusto.params["Selic_lag"],
                       "p-valor (HAC)": robusto.pvalues["Selic_lag"],
                       "R2 ajustado": ols.rsquared_adj, "n": int(ols.nobs)})

    # (e) Subamostra sem pandemia (2005-2019), sem sentido manter Covid=0 sempre
    sub = painel.loc[:"2019-12"].copy()
    sub = sub.drop(columns=["Covid"])
    for k in LAGS_SELIC:
        y = sub["N_MA"].iloc[maior_lag:]
        X = sm.add_constant(sub[["N_MA_lag1", f"Selic_lag{k}", "Ibov_Ret"]].iloc[maior_lag:])
        m = sm.OLS(y, X, missing="drop").fit().get_robustcov_results(cov_type="HAC", maxlags=NW_LAGS)
        params = pd.Series(m.params, index=X.columns)
        pvalues = pd.Series(m.pvalues, index=X.columns)
        linhas.append({"Especificação": f"Subamostra 2005-2019 (sem Covid) | Selic_lag{k}",
                       "coef Selic (HAC)": params[f"Selic_lag{k}"],
                       "p-valor (HAC)": pvalues[f"Selic_lag{k}"],
                       "R2 ajustado": np.nan, "n": int(m.nobs)})

    return pd.DataFrame(linhas)


# =============================================================================
# 5. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 70)
    print("1. DADOS")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final: {len(negocios):,}")

    painel, faltando = montar_painel(negocios)
    painel.to_csv(SAIDA / "painel_mensal.csv")
    print(f"Painel: {painel.index[0]} a {painel.index[-1]} ({len(painel)} meses)")
    print(f"Meses perdidos por defasagem (esperado, não é erro):\n{faltando}")

    desc = estatisticas_descritivas(painel)
    desc.to_csv(SAIDA / "estatisticas_descritivas.csv")
    print("\nEstatísticas descritivas:")
    print(desc.round(2))
    grafico_series(painel)

    print("\n2. RAIZ UNITÁRIA (ADF, Passo 2 do plano de trabalho)")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria.csv", index=False)
    print(ru.round(3).to_string(index=False))
    if not ru["Estacionária a 5%"].all():
        print("AVISO: pelo menos uma variável não é estacionária em nível — considerar "
              "reespecificar em variação/1ª diferença antes da versão final do TCC.")

    print("\n3. MODELOS MQO (mesma amostra, 1 modelo por defasagem da Selic)")
    dados = amostra_comum(painel)
    modelos = {f"Selic_lag{k}": estimar_modelo(dados, k) for k in LAGS_SELIC}
    for nome, (ols, robusto, X, y) in modelos.items():
        with open(SAIDA / f"var_resumo_{nome}.txt", "w", encoding="utf-8") as f:
            f.write(str(ols.summary()))
            f.write("\n\n--- Erros-padrão HAC (Newey-West) ---\n")
            f.write(str(robusto.summary()))

    tabela = tabela_resultados(modelos)
    tabela.to_csv(SAIDA / "resultados_modelos.csv")
    print(tabela.to_string())
    print("\nLegenda: coeficiente*** (p-valor HAC/Newey-West) [IC 95% inferior; superior]; "
          "*** p<0,01, ** p<0,05, * p<0,10.")
    print("\n-> Hipótese principal: ver a linha 'Selic_lag' em cada coluna. Sinal negativo "
          "e significativo = suporte à hipótese de que altas de Selic reduzem o M&A.")

    print("\n3b. COMPARAÇÃO DO COEFICIENTE DA SELIC ENTRE OS 3 MODELOS (item 4 do plano de trabalho)")
    comp_selic = tabela_comparacao_selic(modelos)
    comp_selic.to_csv(SAIDA / "comparacao_selic.csv", index=False)
    print(comp_selic.round(4).to_string(index=False))

    print("\n4. ROBUSTEZ")
    rb = robustez(painel, dados)
    rb.to_csv(SAIDA / "robustez.csv", index=False)
    print(rb.round(4).to_string(index=False))

    print(f"\nPronto. Resultados em: {SAIDA}")


if __name__ == "__main__":
    main()
