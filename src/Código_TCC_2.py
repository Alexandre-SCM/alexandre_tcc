"""
A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
Evidências dos Ciclos da Selic (2005-2024)

Versão 2 — mudanças em relação ao Código_TCC.py:
  - M&A em log, não em nível.
  - Selic em variação (ΔSelic), não em nível (o nível não é estacionário).
  - Busca de especificação: defasagens de 1 a 9 meses + controles extras
    (IPCA, ΔIBC-Br, IIE-Br, vol. do Ibovespa), ranking por BIC.
  - Teste de sazonalidade (dummies de mês) e checagem com Poisson/Binomial
    Negativo (M&A é uma contagem).

Não substitui o Código_TCC.py — fica lado a lado para comparar.
Saídas em ./resultados_2.
"""

from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados_2"
SAIDA.mkdir(exist_ok=True)

DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0
REMOVER_DUPLICATAS = False

INICIO, FIM = "2005-01", "2024-12"

LAGS_SELIC_PRINCIPAIS = [1, 3, 6]     # os 3 modelos centrais do plano de trabalho
LAGS_SELIC_BUSCA = range(1, 10)        # grade estendida só para a busca de robustez

COVID_INICIO, COVID_FIM = "2020-03", "2020-12"
NW_LAGS = 6

CONTROLES_CANDIDATOS = ["IPCA", "dIBC_Br", "IIE_Br", "Ibov_Vol"]

# =============================================================================
# 1. DADOS
# =============================================================================


def carregar_negocios() -> pd.DataFrame:
    df = pd.read_excel(ARQUIVO, sheet_name="Transactions_Results", header=2)
    df["Announcement Date"] = pd.to_datetime(df["Announcement Date"])
    df["valor"] = pd.to_numeric(df["Deal Size (USD, MM)"], errors="coerce")
    df["pct"] = pd.to_numeric(df["% of Target Shares Sought"], errors="coerce")
    return df


def aplicar_filtros(df, deal_min=DEAL_MIN_USD_MM, pct_min=PCT_MIN, dedup=REMOVER_DUPLICATAS):
    m = (df["Deal Status"] == "Completed") & (df["Country (Target/Issuer)"] == "Brazil")
    if deal_min is not None:
        m &= df["valor"] > deal_min
    if pct_min is not None:
        m &= df["pct"] > pct_min
    out = df[m].copy()
    if dedup:
        out = out.drop_duplicates(["Company Name (Target/Issuer)", "Announcement Date", "valor"])
    return out


def contagem_mensal(negocios: pd.DataFrame) -> pd.Series:
    per = negocios["Announcement Date"].dt.to_period("M")
    s = per.value_counts().sort_index()
    idx = pd.period_range(INICIO, FIM, freq="M")
    return s.reindex(idx, fill_value=0).rename("N_MA")


def _serie_macro(m, col_data, col_val, nome):
    s = m.iloc[:, [col_data, col_val]].copy()
    s.columns = ["data", nome]
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    s = s.dropna()
    s.index = s["data"].dt.to_period("M")
    return pd.to_numeric(s[nome], errors="coerce")


def carregar_macro() -> pd.DataFrame:
    m = pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)
    macro = pd.concat([
        _serie_macro(m, 0, 1, "Selic"),
        _serie_macro(m, 7, 8, "IPCA"),
        _serie_macro(m, 13, 14, "IBC_Br"),
        _serie_macro(m, 19, 20, "IIE_Br"),
        _serie_macro(m, 28, 29, "Ibov_Nivel"),
        _serie_macro(m, 28, 30, "Ibov_Vol"),
    ], axis=1)
    macro["Ibov_Ret"] = 100 * np.log(macro["Ibov_Nivel"]).diff()
    return macro


def montar_painel(negocios: pd.DataFrame) -> pd.DataFrame:
    painel = pd.concat([contagem_mensal(negocios), carregar_macro()], axis=1)
    painel = painel.loc[INICIO:FIM]

    # --- variável dependente em log (Grönroos 2024) ---
    painel["log_N_MA"] = np.log(painel["N_MA"])
    painel["log_N_MA_lag1"] = painel["log_N_MA"].shift(1)
    painel["N_MA_lag1"] = painel["N_MA"].shift(1)

    # --- Selic em VARIAÇÃO (não em nível — nível não é estacionário) ---
    painel["dSelic"] = painel["Selic"].diff()
    for k in LAGS_SELIC_BUSCA:
        painel[f"dSelic_lag{k}"] = painel["dSelic"].shift(k - 1)  # variação ocorrida k meses atrás
        painel[f"Selic_lag{k}"] = painel["Selic"].shift(k)         # nível, mantido p/ comparação com a v1

    # --- controle de atividade econômica em variação (IBC-Br não é estacionário em nível) ---
    painel["dIBC_Br"] = painel["IBC_Br"].pct_change() * 100

    # --- dummy Covid ---
    idx_str = painel.index.astype(str)
    painel["Covid"] = ((idx_str >= COVID_INICIO) & (idx_str <= COVID_FIM)).astype(int)

    # --- dummies de mês (sazonalidade) ---
    meses = painel.index.month
    for mnum in range(2, 13):
        painel[f"mes_{mnum}"] = (meses == mnum).astype(int)

    painel.index.name = "Periodo"
    return painel


# =============================================================================
# 2. ESTACIONARIEDADE
# =============================================================================


def testes_raiz_unitaria(painel: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    candidatas = {
        "N_MA (nível)": painel["N_MA"],
        "log(N_MA)": painel["log_N_MA"],
        "Selic (nível)": painel["Selic"],
        "ΔSelic (variação)": painel["dSelic"].dropna(),
        "Ibov_Ret": painel["Ibov_Ret"],
        "IBC_Br (nível)": painel["IBC_Br"],
        "ΔIBC_Br (variação %)": painel["dIBC_Br"].dropna(),
        "IIE_Br (nível)": painel["IIE_Br"],
    }
    for nome, serie in candidatas.items():
        serie = serie.dropna()
        adf = adfuller(serie, regression="c", autolag="AIC")
        linhas.append({"Variável": nome, "ADF stat": adf[0], "ADF p-valor": adf[1],
                       "Estacionária a 5%": adf[1] < .05})
    return pd.DataFrame(linhas)


# =============================================================================
# 3. MODELOS MQO EM LOG (os 3 modelos centrais, agora em log e com ΔSelic)
# =============================================================================


def amostra_comum(painel: pd.DataFrame, lags=LAGS_SELIC_PRINCIPAIS) -> pd.DataFrame:
    maior_lag = max(lags)
    cols = ["N_MA", "N_MA_lag1", "log_N_MA", "log_N_MA_lag1", "Ibov_Ret", "Covid"] + \
        [f"dSelic_lag{k}" for k in lags] + [f"Selic_lag{k}" for k in lags]
    return painel[cols].iloc[maior_lag:].copy()


def _hac(ols, X):
    bruto = ols.get_robustcov_results(cov_type="HAC", maxlags=NW_LAGS)
    ci = pd.DataFrame(bruto.conf_int(alpha=0.05), index=X.columns, columns=["ci_low", "ci_high"])
    return type("HAC", (), {
        "params": pd.Series(bruto.params, index=X.columns),
        "pvalues": pd.Series(bruto.pvalues, index=X.columns),
        "bse": pd.Series(bruto.bse, index=X.columns),
        "conf_int": ci,
        "summary": bruto.summary,
    })()


def estimar_log(dados: pd.DataFrame, lag_selic: int):
    y = dados["log_N_MA"]
    X = dados[["log_N_MA_lag1", f"dSelic_lag{lag_selic}", "Ibov_Ret", "Covid"]].rename(
        columns={f"dSelic_lag{lag_selic}": "dSelic_lag"})
    X = sm.add_constant(X)
    ols = sm.OLS(y, X).fit()
    return ols, _hac(ols, X), X, y


def diagnosticos_modelo(ols, X) -> dict:
    resid = ols.resid
    bp = het_breuschpagan(resid, ols.model.exog)
    jb = jarque_bera(resid)
    dw = durbin_watson(resid)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                    index=X.columns)
    return {
        "R2": ols.rsquared, "R2_ajustado": ols.rsquared_adj, "AIC": ols.aic, "BIC": ols.bic,
        "F p-valor": ols.f_pvalue, "n": int(ols.nobs),
        "Breusch-Pagan p-valor": bp[1], "Jarque-Bera p-valor": jb[1],
        "Durbin-Watson": dw, "VIF máximo (exceto const.)": vif.drop("const").max(),
    }


def tabela_resultados(modelos: dict) -> pd.DataFrame:
    linhas = {}
    for nome, (ols, robusto, X, y) in modelos.items():
        col = {}
        for var in X.columns:
            s, p = robusto.params[var], robusto.pvalues[var]
            ci_low, ci_high = robusto.conf_int.loc[var, "ci_low"], robusto.conf_int.loc[var, "ci_high"]
            estrela = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
            col[var] = f"{s:.4f}{estrela} ({p:.3f}) [{ci_low:.4f}; {ci_high:.4f}]"
        col.update({k: (f"{v:.3f}" if isinstance(v, float) else v)
                   for k, v in diagnosticos_modelo(ols, X).items()})
        linhas[nome] = col
    return pd.DataFrame(linhas)


def tabela_comparacao_dselic(modelos: dict) -> pd.DataFrame:
    """Item 4 do plano de trabalho, aplicado à variável dSelic_lag: comparar
    sinal, magnitude e IC 95% entre os 3 modelos, lado a lado."""
    linhas = []
    for nome, (ols, robusto, X, y) in modelos.items():
        var = "dSelic_lag"
        linhas.append({
            "Modelo": nome,
            "Coeficiente dSelic": robusto.params[var],
            "Erro-padrão (HAC)": robusto.bse[var],
            "p-valor (HAC)": robusto.pvalues[var],
            "IC 95% inferior": robusto.conf_int.loc[var, "ci_low"],
            "IC 95% superior": robusto.conf_int.loc[var, "ci_high"],
            "Significativo a 5%": bool(robusto.pvalues[var] < .05),
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 4. BUSCA DE ESPECIFICAÇÃO (robustez estendida)
# =============================================================================


def busca_especificacao(painel: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada defasagem de 1 a 9 meses e cada subconjunto de controles
    adicionais (até 2 por vez, para manter parcimônia — grau de liberdade
    limitado em 234 obs.), estima log(N_MA) ~ log(N_MA_lag1) + dSelic_lag +
    Ibov_Ret + Covid + controles, e reporta BIC/R² ajustado/coeficiente da
    Selic. Ordenado por BIC (menor = melhor, penaliza excesso de parâmetros).
    """
    maior_lag = max(LAGS_SELIC_BUSCA)
    base_cols = ["log_N_MA", "log_N_MA_lag1", "Ibov_Ret", "Covid"] + \
        [f"dSelic_lag{k}" for k in LAGS_SELIC_BUSCA] + CONTROLES_CANDIDATOS
    d = painel[base_cols].iloc[maior_lag:].copy()

    subconjuntos = [()]
    for r in (1, 2):
        subconjuntos += list(combinations(CONTROLES_CANDIDATOS, r))

    linhas = []
    for k in LAGS_SELIC_BUSCA:
        for controles in subconjuntos:
            cols_x = ["log_N_MA_lag1", f"dSelic_lag{k}", "Ibov_Ret", "Covid"] + list(controles)
            y = d["log_N_MA"]
            X = sm.add_constant(d[cols_x])
            ols = sm.OLS(y, X, missing="drop").fit()
            linhas.append({
                "lag_Selic": k, "controles": ", ".join(controles) or "(nenhum)",
                "coef_dSelic": ols.params[f"dSelic_lag{k}"],
                "p_valor_dSelic": ols.pvalues[f"dSelic_lag{k}"],
                "R2_ajustado": ols.rsquared_adj, "AIC": ols.aic, "BIC": ols.bic,
                "n": int(ols.nobs),
            })
    tab = pd.DataFrame(linhas).sort_values("BIC").reset_index(drop=True)
    return tab


def testar_sazonalidade(painel: pd.DataFrame, lag_selic=3) -> dict:
    """F-teste conjunto das 11 dummies de mês, no modelo principal (log)."""
    maior_lag = max(LAGS_SELIC_PRINCIPAIS)
    d = painel.iloc[maior_lag:].copy()
    dummies = [f"mes_{m}" for m in range(2, 13)]
    y = d["log_N_MA"]
    X_sem = sm.add_constant(d[["log_N_MA_lag1", f"dSelic_lag{lag_selic}", "Ibov_Ret", "Covid"]])
    X_com = sm.add_constant(d[["log_N_MA_lag1", f"dSelic_lag{lag_selic}", "Ibov_Ret", "Covid"] + dummies])
    m_sem = sm.OLS(y, X_sem, missing="drop").fit()
    m_com = sm.OLS(y, X_com, missing="drop").fit()
    f_teste = m_com.compare_f_test(m_sem)
    return {"F": f_teste[0], "p_valor": f_teste[1], "significativo_5pct": f_teste[1] < .05,
           "BIC sem sazonalidade": m_sem.bic, "BIC com sazonalidade": m_com.bic}


# =============================================================================
# 5. CONTAGEM: POISSON / BINOMIAL NEGATIVO (checagem alternativa ao MQO em log)
# =============================================================================


def contagem_alternativa(painel: pd.DataFrame, lag_selic=3) -> pd.DataFrame:
    """
    M&A_t é uma contagem (não-negativa, valores inteiros pequenos) — a família
    Poisson/Binomial Negativo é a alternativa padrão na literatura de contagem
    de eventos (nº de deals) ao MQO em log, e não exige log(0) nem assume
    erros normais. Comparado aqui só como checagem de robustez do sinal e
    significância da Selic, não substitui o MQO pedido no plano de trabalho.
    """
    maior_lag = max(LAGS_SELIC_PRINCIPAIS)
    d = painel.iloc[maior_lag:].copy()
    linhas = []
    for k in LAGS_SELIC_PRINCIPAIS:
        y = d["N_MA"]
        X = sm.add_constant(d[["N_MA_lag1", f"dSelic_lag{k}", "Ibov_Ret", "Covid"]])
        poisson = Poisson(y, X, missing="drop").fit(disp=0)
        # Teste de sobredispersão simples: var(N_MA) vs média(N_MA)
        try:
            binom_neg = NegativeBinomial(y, X, missing="drop").fit(disp=0)
            alpha = binom_neg.params.get("alpha", np.nan)
        except Exception:
            binom_neg, alpha = None, np.nan
        linhas.append({
            "lag_Selic": k,
            "Poisson: coef dSelic": poisson.params[f"dSelic_lag{k}"],
            "Poisson: p-valor": poisson.pvalues[f"dSelic_lag{k}"],
            "Poisson: AIC": poisson.aic,
            "BinomNeg: coef dSelic": binom_neg.params[f"dSelic_lag{k}"] if binom_neg is not None else np.nan,
            "BinomNeg: p-valor": binom_neg.pvalues[f"dSelic_lag{k}"] if binom_neg is not None else np.nan,
            "BinomNeg: AIC": binom_neg.aic if binom_neg is not None else np.nan,
            "dispersão (var/média de N_MA)": d["N_MA"].var() / d["N_MA"].mean(),
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 6. GRÁFICOS
# =============================================================================


def grafico_comparativo(painel: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(painel.index.to_timestamp(), painel["N_MA"], color="#1f4e79", label="N_MA (nível)")
    ax[0].set_title("Número mensal de anúncios de M&A (nível)")
    ax[0].axvspan(pd.Period(COVID_INICIO).to_timestamp(), pd.Period(COVID_FIM).to_timestamp(),
                  color="gray", alpha=.2)
    ax[1].plot(painel.index.to_timestamp(), painel["log_N_MA"], color="#2e7d32", label="log(N_MA)")
    ax[1].set_title("log do número mensal de anúncios de M&A")
    fig.tight_layout()
    fig.savefig(SAIDA / "log_vs_nivel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(painel.index.to_timestamp(), painel["dSelic"], color="#b7472a")
    ax.axhline(0, color="k", lw=.7)
    ax.set_title("Variação mensal da Selic (p.p.) — variável usada nos modelos v2")
    fig.tight_layout()
    fig.savefig(SAIDA / "variacao_selic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 70)
    print("1. DADOS")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final: {len(negocios):,}")

    painel = montar_painel(negocios)
    painel.to_csv(SAIDA / "painel_mensal_v2.csv")
    grafico_comparativo(painel)

    print("\n2. RAIZ UNITÁRIA — nível vs. transformações usadas na v2")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria_v2.csv", index=False)
    print(ru.round(3).to_string(index=False))
    print("\n-> ΔSelic e ΔIBC-Br resolvem a não estacionariedade dos níveis correspondentes.")

    print("\n3. MODELOS PRINCIPAIS EM LOG (mesma estrutura do plano de trabalho, "
          "log(M&A) e ΔSelic no lugar do nível)")
    dados = amostra_comum(painel)
    modelos = {f"log | dSelic_lag{k}": estimar_log(dados, k) for k in LAGS_SELIC_PRINCIPAIS}
    for nome, (ols, robusto, X, y) in modelos.items():
        arq = nome.replace(" | ", "_").replace(" ", "")
        with open(SAIDA / f"resumo_{arq}.txt", "w", encoding="utf-8") as f:
            f.write(str(ols.summary()))
            f.write("\n\n--- Erros-padrão HAC (Newey-West) ---\n")
            f.write(str(robusto.summary()))
    tabela = tabela_resultados(modelos)
    tabela.to_csv(SAIDA / "modelos_log_principais.csv")
    print(tabela.to_string())
    print("\nLegenda: coeficiente*** (p-valor HAC/Newey-West) [IC 95% inferior; superior]; "
          "*** p<0,01, ** p<0,05, * p<0,10.")
    print("Coeficiente de dSelic_lag em log(M&A) ~ variação percentual de M&A para um "
          "choque de +1 p.p. na Selic k meses antes.")

    print("\n3b. COMPARAÇÃO DO COEFICIENTE DE dSelic ENTRE OS 3 MODELOS (item 4 do plano de trabalho)")
    comp_dselic = tabela_comparacao_dselic(modelos)
    comp_dselic.to_csv(SAIDA / "comparacao_dselic.csv", index=False)
    print(comp_dselic.round(4).to_string(index=False))

    print("\n4. COMPARAÇÃO COM A V1 (nível, Selic em nível) — mesmos meses, mesmo controle")
    modelos_nivel = {}
    for k in LAGS_SELIC_PRINCIPAIS:
        y = dados["N_MA"]
        X = sm.add_constant(dados[["N_MA_lag1", f"Selic_lag{k}", "Ibov_Ret", "Covid"]])
        ols = sm.OLS(y, X, missing="drop").fit()
        modelos_nivel[f"nível | Selic_lag{k}"] = ols
    comp = pd.DataFrame({
        nome: {"R2_ajustado": m.rsquared_adj, "AIC": m.aic, "BIC": m.bic,
              "JB p-valor": jarque_bera(m.resid)[1],
              "BP p-valor": het_breuschpagan(m.resid, m.model.exog)[1]}
        for nome, m in modelos_nivel.items()
    })
    comp_log = pd.DataFrame({
        nome: {"R2_ajustado": ols.rsquared_adj, "AIC": ols.aic, "BIC": ols.bic,
              "JB p-valor": jarque_bera(ols.resid)[1],
              "BP p-valor": het_breuschpagan(ols.resid, ols.model.exog)[1]}
        for nome, (ols, *_r) in modelos.items()
    })
    diagnostico_comp = pd.concat([comp, comp_log], axis=1)
    diagnostico_comp.to_csv(SAIDA / "diagnosticos_comparativos.csv")
    print(diagnostico_comp.round(3).to_string())
    print("\n(AIC/BIC não são comparáveis 1-a-1 entre nível e log — a comparação correta é "
          "JB/BP, que mostram se os resíduos de cada versão passam nos testes de "
          "normalidade/homocedasticidade.)")

    print("\n5. BUSCA DE ESPECIFICAÇÃO (lag 1-9, controles extras, ranking por BIC)")
    busca = busca_especificacao(painel)
    busca.to_csv(SAIDA / "busca_especificacao.csv", index=False)
    print(busca.head(15).round(4).to_string(index=False))
    melhor = busca.iloc[0]
    print(f"\nMelhor especificação por BIC: lag={melhor['lag_Selic']}, "
          f"controles=[{melhor['controles']}], coef_dSelic={melhor['coef_dSelic']:.4f} "
          f"(p={melhor['p_valor_dSelic']:.3f})")

    print("\n6. SAZONALIDADE (dummies de mês, F-teste conjunto)")
    saz = testar_sazonalidade(painel)
    pd.Series(saz).to_csv(SAIDA / "teste_sazonalidade.csv")
    print(saz)
    if saz["significativo_5pct"]:
        print("-> Sazonalidade mensal é conjuntamente significativa: considerar manter as "
              "dummies de mês no modelo final.")
    else:
        print("-> Sazonalidade mensal não é conjuntamente significativa: dispensável, mantém "
              "o modelo mais parcimonioso.")

    print("\n7. CHECAGEM ALTERNATIVA: POISSON / BINOMIAL NEGATIVO (M&A é contagem)")
    cont = contagem_alternativa(painel)
    cont.to_csv(SAIDA / "poisson_binomial_negativo.csv", index=False)
    print(cont.round(4).to_string(index=False))
    disp = cont["dispersão (var/média de N_MA)"].iloc[0]
    print(f"\nRazão variância/média de N_MA = {disp:.2f}. "
          + ("> 1: há sobredispersão — o Binomial Negativo é mais adequado que o Poisson."
             if disp > 1.2 else "≈ 1: Poisson é adequado."))

    print(f"\nPronto. Resultados em: {SAIDA}")


if __name__ == "__main__":
    main()
