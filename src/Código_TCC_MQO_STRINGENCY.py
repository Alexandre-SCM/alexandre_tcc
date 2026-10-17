"""
MQO mensal em log com o ÍNDICE DE RIGOR (Stringency Index, OxCGRT) no lugar da dummy da Covid
==============================================================================================

Fonte: Oxford COVID-19 Government Response Tracker (OxCGRT), arquivo Stringency_index.csv
(separador ';'; segunda linha são tags HXL). Usa-se a coluna StringencyIndex_Average do Brasil,
Jurisdiction == NAT_TOTAL, diária de 2020-01-01 a 2022-12-31, escala 0-100.

Tratamento:
  - média mensal do índice, dividida por 100 (coeficiente = efeito de ir de "sem restrição" a
    "rigor máximo");
  - antes de 2020 = 0 (sem restrição);
  - depois de 2022-12 o OxCGRT deixou de atualizar a série: por padrão repete-se o último valor
    (22,2) em 2023-24; a sensibilidade 'zero em 2023+' está nos resultados.

Modelos (todas as variáveis em log como em Código_TCC_MQO_LOG.py; Meta_log, ΔMeta_log, HAC 6):
  S0  controles com a dummy Covid (mar-dez/2020)   <- base de comparação
  S1  Stringency no lugar da dummy
  S2  Stringency em t e t-1 (efeito defasado)
  S3  dummy + Stringency juntos
  S4  S1 com 2023+ = 0
  Para L1 (ΔMeta_log, soma 0-3) e L2 (Meta_log, k=3). Também: grade k=0..12 de ΔMeta_log com
  Stringency, subamostras e exclusão de 2020-2022 (a fragilidade do efeito positivo da variação).

Rodar: python Código_TCC_MQO_STRINGENCY.py   (precisa de Stringency_index.csv na pasta do script
ou em ./dados_covid/; gera ./dados_covid/stringency_brasil_mensal.csv e ./resultados_mqo_stringency)
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_mqo_stringency"
DIR_COVID = BASE / "dados_covid"
SAIDA.mkdir(exist_ok=True)
DIR_COVID.mkdir(exist_ok=True)
CACHE_MENSAL = DIR_COVID / "stringency_brasil_mensal.csv"
CANDIDATOS = [BASE / "Stringency_index.csv", DIR_COVID / "Stringency_index.csv"]

spec = importlib.util.spec_from_file_location("lg", BASE / "Código_TCC_MQO_LOG.py")
lg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lg)
mqo = lg.mqo


# ----------------------------------------------------------------------------
def stringency_mensal() -> pd.Series:
    if CACHE_MENSAL.exists():
        s = pd.read_csv(CACHE_MENSAL, index_col=0)["stringency"]
        s.index = pd.PeriodIndex(s.index, freq="M")
        return s
    arq = next((a for a in CANDIDATOS if a.exists()), None)
    if arq is None:
        raise SystemExit("Coloque Stringency_index.csv em " + " ou ".join(str(a.parent) for a in CANDIDATOS))
    d = pd.read_csv(arq, sep=";", skiprows=[1], dtype=str,
                    usecols=["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average"])
    b = d[(d.CountryCode == "BRA") & (d.Jurisdiction == "NAT_TOTAL")].copy()
    b["Date"] = pd.to_datetime(b["Date"], format="%Y%m%d")
    b["x"] = pd.to_numeric(b["StringencyIndex_Average"])
    s = b.set_index("Date")["x"].resample("ME").mean()
    s.index = s.index.to_period("M")
    s.rename("stringency").to_frame().to_csv(CACHE_MENSAL)
    return s


def juntar(p: pd.DataFrame) -> pd.DataFrame:
    s = stringency_mensal() / 100.0
    p = p.copy()
    p["Str"] = s.reindex(p.index)
    ate = s.index.max()
    p.loc[p.index < s.index.min(), "Str"] = 0.0
    p["Str_zero23"] = p["Str"]
    p.loc[p.index > ate, "Str"] = s.iloc[-1]          # padrão: repete o último valor
    p.loc[p.index > ate, "Str_zero23"] = 0.0           # sensibilidade
    p["Str_l1"] = p["Str"].shift(1)
    return p


POL1 = [f"dMetaLog_l{k}" for k in range(4)]
BASE_CTRL = ["Ibov_Ret", "dIBC_log"]


def ajustar(p, xs, pol, ini="padrao", **kw):
    hac, plain, d = lg.estimar(p, xs, ini)
    if len(pol) > 1:
        b, ep, pv = lg.soma(hac, pol)
    else:
        b, ep, pv = hac.params[pol[0]], hac.bse[pol[0]], hac.pvalues[pol[0]]
    bg = acorr_breusch_godfrey(plain, nlags=6)[1]
    jb = jarque_bera(plain.resid)[1]
    bp = het_breuschpagan(plain.resid, plain.model.exog)[1]
    return hac, {"coef-chave": b, "ep_HAC": ep, "p": pv, "n": int(hac.nobs), "R2_aj": hac.rsquared_adj,
                 "BG(6) p": bg, "JB p": jb, "BP p": bp, "AIC": hac.aic, "BIC": hac.bic}


def tabela_modelos(p):
    esp = {
        "S0 dummy Covid": (["Covid"], "Covid"),
        "S1 Stringency": (["Str"], "Str"),
        "S2 Stringency t e t-1": (["Str", "Str_l1"], "Str"),
        "S3 dummy + Stringency": (["Covid", "Str"], "Str"),
        "S4 Stringency (2023+ = 0)": (["Str_zero23"], "Str_zero23"),
    }
    linhas = []
    for fam, pol, base in (("L1 ΔMeta_log 0-3", POL1, ["log_N_MA_lag1"] + POL1),
                           ("L2 Meta_log k=3", ["MetaLog_l3"], ["log_N_MA_lag1", "MetaLog_l3"])):
        for nome, (cv, foco) in esp.items():
            xs = base + BASE_CTRL + cv
            hac, r = ajustar(p, xs, pol)
            r.update({"família": fam, "controle de Covid": nome,
                      "coef Covid/Str": hac.params[foco], "p Covid/Str": hac.pvalues[foco]})
            if "Str_l1" in cv:
                b, ep, pv = lg.soma(hac, ["Str", "Str_l1"])
                r.update({"coef Covid/Str": b, "p Covid/Str": pv})
            linhas.append(r)
    t = pd.DataFrame(linhas)
    cols = ["família", "controle de Covid", "coef-chave", "ep_HAC", "p", "coef Covid/Str", "p Covid/Str", "n", "R2_aj", "AIC", "BIC",
            "BG(6) p", "JB p", "BP p"]
    return t[cols]


def grade(p, cv):
    linhas = []
    for k in range(mqo.MAX_LAG + 1):
        xs = ["log_N_MA_lag1", f"dMetaLog_l{k}"] + BASE_CTRL + cv
        hac, plain, d = lg.estimar(p, xs)
        linhas.append({"k_meses": k, **lg.coef_linha(hac, f"dMetaLog_l{k}"), "n": int(hac.nobs)})
    g = pd.DataFrame(linhas)
    g["p_Holm"] = multipletests(g["p_valor"], method="holm")[1]
    return g


def nivel(p, cv):
    linhas = []
    for k in (1, 3, 6, 12):
        xs = ["log_N_MA_lag1", f"MetaLog_l{k}"] + BASE_CTRL + cv
        hac, plain, d = lg.estimar(p, xs)
        linhas.append({"k_meses": k, **lg.coef_linha(hac, f"MetaLog_l{k}"), "n": int(hac.nobs)})
    return pd.DataFrame(linhas)


def subamostras(p, cv):
    linhas = []
    cortes = {"Completo": None, "Sem 2020-2022": [2020, 2021, 2022], "Até 2019": "ate2019", "2015-2024": "2015"}
    for nome, c in cortes.items():
        for fam, pol, base in (("L1 ΔMeta_log 0-3", POL1, ["log_N_MA_lag1"] + POL1),
                               ("L2 Meta_log k=3", ["MetaLog_l3"], ["log_N_MA_lag1", "MetaLog_l3"])):
            xs = base + BASE_CTRL + cv
            if isinstance(c, list):
                d = mqo.recorte(p, lg.INI_PADRAO, mqo.FIM)
                d = d[~d.index.year.isin(c)]
                xs = [x for x in xs if x not in ("Covid", "Str", "Str_zero23", "Str_l1")]   # constantes/irrelevantes sem 2020-22
            elif c == "ate2019":
                d = mqo.recorte(p, lg.INI_PADRAO, "2019-12")
                xs = [x for x in xs if x not in ("Covid", "Str", "Str_zero23", "Str_l1")]
            elif c == "2015":
                d = mqo.recorte(p, "2015-01", mqo.FIM)
            else:
                d = mqo.recorte(p, lg.INI_PADRAO, mqo.FIM)
            d = d.dropna(subset=["log_N_MA"] + xs)
            hac, plain = mqo.ols_hac(d["log_N_MA"], d[xs])
            if len(pol) > 1:
                b, ep, pv = lg.soma(hac, pol)
            else:
                b, ep, pv = hac.params[pol[0]], hac.bse[pol[0]], hac.pvalues[pol[0]]
            linhas.append({"família": fam, "amostra": nome, "coef-chave": b, "ep_HAC": ep, "p": pv, "n": int(hac.nobs)})
    return pd.DataFrame(linhas)


def grafico(p, g0, g1):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    x = p.loc["2019-06":"2023-12"]
    ax[0].plot(x.index.to_timestamp(), x["Str"] * 100, color="#1f3864", label="Stringency (mensal)")
    ax[0].fill_between(x.index.to_timestamp(), 0, 100 * x["Covid"], color="#d62728", alpha=0.18, label="dummy Covid (mar-dez/2020)")
    ax[0].set_ylabel("Índice de rigor (0-100)")
    ax[0].set_title("Stringency Index (Brasil) x dummy Covid")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)
    ax[1].plot(g0["k_meses"], g0["coef"], "o-", color="#7f7f7f", label="controle: dummy Covid")
    ax[1].plot(g1["k_meses"], g1["coef"], "s-", color="#1f3864", label="controle: Stringency")
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("Defasagem k (meses)")
    ax[1].set_ylabel("Coef. de ΔMeta_log")
    ax[1].set_title("Efeito de ΔMeta por defasagem")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "stringency_e_grade.png")
    plt.close(fig)


def main():
    neg = mqo.aplicar_filtros(mqo.carregar_negocios())
    p = juntar(lg.construir(mqo.montar_painel(neg)))
    pd.set_option("display.width", 220)

    t = tabela_modelos(p)
    t.to_csv(SAIDA / "modelos_covid_dummy_vs_stringency.csv", index=False)
    print("== DUMMY x STRINGENCY ==")
    print(t.round(4).to_string())

    g0, g1 = grade(p, ["Covid"]), grade(p, ["Str"])
    g0.to_csv(SAIDA / "grade_dMeta_log_dummy.csv", index=False)
    g1.to_csv(SAIDA / "grade_dMeta_log_stringency.csv", index=False)
    print("\n== GRADE com Stringency ==")
    print(g1.round(4).to_string())

    n0, n1 = nivel(p, ["Covid"]), nivel(p, ["Str"])
    n0.to_csv(SAIDA / "nivel_dummy.csv", index=False)
    n1.to_csv(SAIDA / "nivel_stringency.csv", index=False)
    print("\n== NÍVEL com Stringency ==")
    print(n1.round(4).to_string())

    s0, s1 = subamostras(p, ["Covid"]), subamostras(p, ["Str"])
    s0["controle"], s1["controle"] = "dummy", "Stringency"
    ss = pd.concat([s0, s1])
    ss.to_csv(SAIDA / "subamostras.csv", index=False)
    print("\n== SUBAMOSTRAS ==")
    print(ss.round(4).to_string())

    p.loc[mqo.INICIO:mqo.FIM, ["Covid", "Str", "Str_zero23"]].to_csv(SAIDA / "painel_stringency.csv")
    grafico(p, g0, g1)
    print("\nPronto. Saída em", SAIDA)


if __name__ == "__main__":
    main()
