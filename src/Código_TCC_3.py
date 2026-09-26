"""
A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
Evidências dos Ciclos da Selic (2005-2024)

Versão 3 — mesma base e mesma pergunta, painel agregado em SEMESTRES em vez
de meses (2005-2024 → 40 semestres). Defasagens de 1, 2 e 3 semestres.

Agregação mês -> semestre: N_MA e IPCA somados (fluxos); Selic, IBC-Br,
IIE-Br e nível do Ibovespa em média (índices/níveis); retorno do Ibovespa
somado (log-retornos são aditivos); Covid = 1 nos dois semestres de 2020.

Roda as duas famílias da v1/v2 (nível; log + ΔSelic). Amostra pequena
(~37 obs. após defasagens) — testes estatísticos têm menos poder.
Saídas em ./resultados_3.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
SAIDA = BASE / "resultados_3"
SAIDA.mkdir(exist_ok=True)

DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0
REMOVER_DUPLICATAS = False

INICIO, FIM = "2005-01", "2024-12"

LAGS_SELIC = [1, 2, 3]     # semestres de defasagem (≈ 6, 12 e 18 meses)
NW_LAGS_SEM = 1            # HAC com poucas defasagens — amostra pequena (~40 semestres)

COVID_ANO = 2020  # dummy = 1 nos dois semestres de 2020

# =============================================================================
# 1. DADOS MENSAIS (mesma lógica das versões 1 e 2)
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


def montar_painel_mensal(negocios: pd.DataFrame) -> pd.DataFrame:
    painel = pd.concat([contagem_mensal(negocios), carregar_macro()], axis=1)
    return painel.loc[INICIO:FIM]


# =============================================================================
# 2. AGREGAÇÃO MENSAL -> SEMESTRAL
# =============================================================================


def rotulo_semestre(idx_mensal: pd.PeriodIndex) -> pd.Index:
    """'2005S1', '2005S2', ... — 1º semestre = jan-jun, 2º semestre = jul-dez."""
    ano = idx_mensal.year
    sem = np.where(idx_mensal.month <= 6, 1, 2)
    return pd.Index([f"{a}S{s}" for a, s in zip(ano, sem)], name="Semestre")


def montar_painel_semestral(painel_mensal: pd.DataFrame) -> pd.DataFrame:
    df = painel_mensal.copy()
    df["Semestre"] = rotulo_semestre(df.index)

    soma = df.groupby("Semestre")[["N_MA", "IPCA"]].sum()
    soma_ret = df.groupby("Semestre")[["Ibov_Ret"]].sum()  # log-retornos são aditivos
    media = df.groupby("Semestre")[["Selic", "IBC_Br", "IIE_Br", "Ibov_Nivel", "Ibov_Vol"]].mean()

    painel = pd.concat([soma, soma_ret, media], axis=1)
    # ordena cronologicamente (o groupby por string ordena alfabeticamente, o
    # que já bate com a ordem cronológica no formato AAAAS{1,2}, mas deixamos
    # explícito por segurança)
    painel = painel.sort_index()

    painel["log_N_MA"] = np.log(painel["N_MA"])
    painel["log_N_MA_lag1"] = painel["log_N_MA"].shift(1)
    painel["N_MA_lag1"] = painel["N_MA"].shift(1)
    painel["dSelic"] = painel["Selic"].diff()
    for k in LAGS_SELIC:
        painel[f"Selic_lag{k}"] = painel["Selic"].shift(k)
        painel[f"dSelic_lag{k}"] = painel["dSelic"].shift(k - 1)

    painel["Covid"] = painel.index.str.startswith(str(COVID_ANO)).astype(int)
    painel.index.name = "Semestre"
    return painel


# =============================================================================
# 3. ESTACIONARIEDADE (com ressalva sobre poder do teste em amostra pequena)
# =============================================================================


def testes_raiz_unitaria(painel: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    candidatas = {
        "N_MA (nível)": painel["N_MA"],
        "log(N_MA)": painel["log_N_MA"],
        "Selic (nível)": painel["Selic"],
        "ΔSelic (variação)": painel["dSelic"].dropna(),
        "Ibov_Ret (semestral)": painel["Ibov_Ret"],
    }
    for nome, serie in candidatas.items():
        serie = serie.dropna()
        # maxlag baixo: amostra pequena não sustenta muitas defasagens no ADF
        adf = adfuller(serie, regression="c", autolag="AIC", maxlag=4)
        linhas.append({"Variável": nome, "ADF stat": adf[0], "ADF p-valor": adf[1],
                       "n": len(serie), "Estacionária a 5%": adf[1] < .05})
    return pd.DataFrame(linhas)


# =============================================================================
# 4. MODELOS MQO — SEMESTRAL
# =============================================================================


def amostra_comum(painel: pd.DataFrame) -> pd.DataFrame:
    maior_lag = max(LAGS_SELIC)
    cols = ["N_MA", "N_MA_lag1", "log_N_MA", "log_N_MA_lag1", "Ibov_Ret", "Covid"] + \
        [f"Selic_lag{k}" for k in LAGS_SELIC] + [f"dSelic_lag{k}" for k in LAGS_SELIC]
    return painel[cols].iloc[maior_lag:].copy()


def _hac(ols, X, nw_lags=NW_LAGS_SEM):
    bruto = ols.get_robustcov_results(cov_type="HAC", maxlags=nw_lags)
    ci = pd.DataFrame(bruto.conf_int(alpha=0.05), index=X.columns, columns=["ci_low", "ci_high"])
    return type("HAC", (), {
        "params": pd.Series(bruto.params, index=X.columns),
        "pvalues": pd.Series(bruto.pvalues, index=X.columns),
        "bse": pd.Series(bruto.bse, index=X.columns),
        "conf_int": ci,
        "summary": bruto.summary,
    })()


def estimar_nivel(dados: pd.DataFrame, lag_selic: int):
    y = dados["N_MA"]
    X = dados[["N_MA_lag1", f"Selic_lag{lag_selic}", "Ibov_Ret", "Covid"]].rename(
        columns={f"Selic_lag{lag_selic}": "Selic_lag"})
    X = sm.add_constant(X)
    ols = sm.OLS(y, X, missing="drop").fit()
    return ols, _hac(ols, X), X, y


def estimar_log(dados: pd.DataFrame, lag_selic: int):
    y = dados["log_N_MA"]
    X = dados[["log_N_MA_lag1", f"dSelic_lag{lag_selic}", "Ibov_Ret", "Covid"]].rename(
        columns={f"dSelic_lag{lag_selic}": "dSelic_lag"})
    X = sm.add_constant(X)
    ols = sm.OLS(y, X, missing="drop").fit()
    return ols, _hac(ols, X), X, y


def diagnosticos_modelo(ols, X) -> dict:
    resid = ols.resid
    bp = het_breuschpagan(resid, ols.model.exog)
    jb = jarque_bera(resid)
    dw = durbin_watson(resid)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                    index=X.columns)
    return {
        "R2": ols.rsquared, "R2_ajustado": ols.rsquared_adj,
        "F p-valor": ols.f_pvalue, "n": int(ols.nobs),
        "Breusch-Pagan p-valor": bp[1], "Jarque-Bera p-valor": jb[1],
        "Durbin-Watson": dw, "VIF máximo (exceto const.)": vif.drop("const").max(),
    }


def tabela_resultados(modelos: dict, var_selic="Selic_lag") -> pd.DataFrame:
    linhas = {}
    for nome, (ols, robusto, X, y) in modelos.items():
        col = {}
        for var in X.columns:
            s, pv = robusto.params[var], robusto.pvalues[var]
            ci_low, ci_high = robusto.conf_int.loc[var, "ci_low"], robusto.conf_int.loc[var, "ci_high"]
            estrela = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .10 else ""
            col[var] = f"{s:.4f}{estrela} ({pv:.3f}) [{ci_low:.4f}; {ci_high:.4f}]"
        col.update({k: (f"{v:.3f}" if isinstance(v, float) else v)
                   for k, v in diagnosticos_modelo(ols, X).items()})
        linhas[nome] = col
    return pd.DataFrame(linhas)


def tabela_comparacao(modelos: dict, var: str) -> pd.DataFrame:
    linhas = []
    for nome, (ols, robusto, X, y) in modelos.items():
        linhas.append({
            "Modelo": nome,
            "Coeficiente": robusto.params[var],
            "Erro-padrão (HAC)": robusto.bse[var],
            "p-valor (HAC)": robusto.pvalues[var],
            "IC 95% inferior": robusto.conf_int.loc[var, "ci_low"],
            "IC 95% superior": robusto.conf_int.loc[var, "ci_high"],
            "Significativo a 5%": bool(robusto.pvalues[var] < .05),
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 5. GRÁFICOS
# =============================================================================


def grafico_semestral(painel: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = range(len(painel))
    ax[0].bar(x, painel["N_MA"], color="#1f4e79")
    ax[0].set_xticks(x[::2])
    ax[0].set_xticklabels(painel.index[::2], rotation=90, fontsize=7)
    ax[0].set_title("Número de anúncios de M&A por semestre (soma)")
    ax[1].plot(x, painel["Selic"], color="#b7472a", marker="o", markersize=3)
    ax[1].set_xticks(x[::2])
    ax[1].set_xticklabels(painel.index[::2], rotation=90, fontsize=7)
    ax[1].set_title("Selic média do semestre (% a.m.)")
    fig.tight_layout()
    fig.savefig(SAIDA / "series_semestrais.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 70)
    print("1. DADOS — agregação mensal -> semestral")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final: {len(negocios):,}")

    painel_mensal = montar_painel_mensal(negocios)
    painel = montar_painel_semestral(painel_mensal)
    painel.to_csv(SAIDA / "painel_semestral.csv")
    print(f"Painel semestral: {painel.index[0]} a {painel.index[-1]} "
          f"({len(painel)} semestres — equivalente a {len(painel_mensal)} meses)")
    print(painel[["N_MA", "Selic", "Ibov_Ret", "Covid"]].describe().round(2))
    grafico_semestral(painel)
    print("\nAVISO: com ~40 semestres (e ~37 após as defasagens), os testes assintóticos "
          "abaixo (ADF, HAC, Jarque-Bera, Breusch-Pagan) têm poder estatístico bem menor "
          "do que nas versões mensais (234 obs.) — leia os resultados como indicativos.")

    print("\n2. RAIZ UNITÁRIA (ADF, amostra semestral)")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria_semestral.csv", index=False)
    print(ru.round(3).to_string(index=False))

    dados = amostra_comum(painel)
    print(f"\nAmostra comum aos 3 modelos: {len(dados)} semestres "
          f"(perde os {max(LAGS_SELIC)} primeiros semestres, pela maior defasagem usada)")

    print("\n3. MODELOS EM NÍVEL (M&A e Selic em nível — réplica direta do plano de "
          "trabalho, só que em semestres)")
    modelos_nivel = {f"nível | Selic_lag{k}": estimar_nivel(dados, k) for k in LAGS_SELIC}
    tab_nivel = tabela_resultados(modelos_nivel, "Selic_lag")
    tab_nivel.to_csv(SAIDA / "modelos_nivel_semestral.csv")
    print(tab_nivel.to_string())

    comp_nivel = tabela_comparacao(modelos_nivel, "Selic_lag")
    comp_nivel.to_csv(SAIDA / "comparacao_selic_semestral.csv", index=False)
    print("\nComparação do coeficiente da Selic entre os 3 modelos (nível):")
    print(comp_nivel.round(4).to_string(index=False))

    print("\n4. MODELOS EM LOG + ΔSelic (especificação recomendada na v2, agora em "
          "semestres)")
    modelos_log = {f"log | dSelic_lag{k}": estimar_log(dados, k) for k in LAGS_SELIC}
    tab_log = tabela_resultados(modelos_log, "dSelic_lag")
    tab_log.to_csv(SAIDA / "modelos_log_semestral.csv")
    print(tab_log.to_string())

    comp_log = tabela_comparacao(modelos_log, "dSelic_lag")
    comp_log.to_csv(SAIDA / "comparacao_dselic_semestral.csv", index=False)
    print("\nComparação do coeficiente de ΔSelic entre os 3 modelos (log):")
    print(comp_log.round(4).to_string(index=False))

    for nome, (ols, robusto, X, y) in {**modelos_nivel, **modelos_log}.items():
        arq = nome.replace(" | ", "_").replace(" ", "")
        with open(SAIDA / f"resumo_{arq}.txt", "w", encoding="utf-8") as f:
            f.write(str(ols.summary()))
            f.write("\n\n--- Erros-padrão HAC (Newey-West) ---\n")
            f.write(str(robusto.summary()))

    print("\n5. COMPARAÇÃO MENSAL x SEMESTRAL (mesmo tipo de modelo, escalas de tempo diferentes)")
    linhas = []
    for k in LAGS_SELIC:
        for fam, modelos_, var in [("nível", modelos_nivel, "Selic_lag"),
                                   ("log+ΔSelic", modelos_log, "dSelic_lag")]:
            nome = f"{'nível' if fam=='nível' else 'log'} | {var}{k}"
            ols, robusto, X, y = modelos_[nome]
            linhas.append({
                "Família": fam, "Defasagem (semestres)": k,
                "≈ meses equivalentes": k * 6,
                "Coeficiente": robusto.params[var], "p-valor (HAC)": robusto.pvalues[var],
                "Significativo a 5%": bool(robusto.pvalues[var] < .05),
                "R2 ajustado": ols.rsquared_adj, "n": int(ols.nobs),
            })
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(SAIDA / "resumo_final_semestral.csv", index=False)
    print(resumo.round(4).to_string(index=False))
    print("\n-> Comparar esta tabela com resultados_modelos.csv (v1, mensal) e "
          "modelos_log_principais.csv (v2, mensal) para ver se o efeito de curtíssimo "
          "prazo encontrado nas versões mensais se sustenta, se dilui ou desaparece "
          "quando o tempo é medido em semestres.")

    print(f"\nPronto. Resultados em: {SAIDA}")


if __name__ == "__main__":
    main()
