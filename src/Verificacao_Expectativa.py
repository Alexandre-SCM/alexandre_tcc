"""
Verificação da variável de expectativa da Selic (Focus) usada em Código_TCC_MQO_LOG.py
=====================================================================================
Responde: (1) a variável está construída certo? (2) os resultados dependem da escolha?
(3) há sinais de endogeneidade/colinearidade?  Saída em ./resultados_verificacao
"""
import importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_verificacao"; SAIDA.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("lg", BASE / "Código_TCC_MQO_LOG.py")
lg = importlib.util.module_from_spec(spec); spec.loader.exec_module(lg)
mqo = lg.mqo

def pinfo(x):
    return f"{x:.4f}"

def main():
    neg = mqo.aplicar_filtros(mqo.carregar_negocios())
    p = lg.construir(mqo.montar_painel(neg))
    pol1 = [f"dMetaLog_l{k}" for k in range(4)]
    C = lg.CTRL_LOG
    out = []

    # ---------- (A) construção ----------
    f = pd.read_csv(lg.CACHE_FOCUS, parse_dates=["Data"])
    md = f.groupby("Reuniao")["Data"].max().sort_values()
    md = md[md < "2026-06-01"]
    meta_d = pd.read_csv(BASE / "dados_focus" / "selic_meta_sgs432.csv", parse_dates=["Data"]).set_index("Data")["Selic_decidida"]
    chg = meta_d[meta_d.diff().fillna(0) != 0].index
    dias = [(chg[chg > d][0] - d).days if (chg > d).any() else np.nan for d in md]
    a = {"reuniões": len(md), "por ano (mín-máx)": f"{md.groupby(md.dt.year).size().min()}-{md.groupby(md.dt.year).size().max()}",
         "intervalo mediano entre reuniões (dias)": float(md.diff().dt.days.median()),
         "reuniões c/ mudança da Meta 1 dia depois da data-proxy": int((np.array(dias) == 1).sum()),
         "reuniões c/ mudança em >3 dias (sem mudança nesse ciclo)": int((np.array(dias) > 3).sum())}
    pd.Series(a).to_csv(SAIDA / "A_construcao.csv")
    print("A. Construção:", a)

    # ---------- (B) definição alternativa da variável ----------
    p["Esp_d6m_ort"] = np.nan
    base = p[["Esp_d6m", "dMeta_log", "Meta_log"]].dropna()
    r = sm.OLS(base["Esp_d6m"], sm.add_constant(base[["dMeta_log", "Meta_log"]])).fit()
    p.loc[base.index, "Esp_d6m_ort"] = r.resid          # parte da expectativa não explicada por ΔMeta e nível
    p["E6_nivel_dev"] = p["E_6m_log"] - p["E_6m_log"].rolling(12).mean()
    alts = {
        "Esp_d6m (usada)": "Esp_d6m", "Esp_dprox (próx. reunião)": "Esp_dprox", "Esp_d12m": "Esp_d12m",
        "Esp_d6m ortogonalizada a ΔMeta e Meta": "Esp_d6m_ort",
    }
    for rot, v in alts.items():
        xs = ["log_N_MA_lag1"] + pol1 + [v] + C
        hac, plain, d = lg.estimar(p, xs)
        b, ep, pv = lg.soma(hac, pol1)
        out.append({"bloco": "B. Definição da expectativa", "especificação": f"L1 + {rot}", "coef Esp": hac.params[v], "p Esp": hac.pvalues[v],
                    "soma ΔMeta 0-3": b, "p soma ΔMeta": pv, "n": int(hac.nobs)})
    for rot, v in alts.items():
        xs = ["log_N_MA_lag1", "MetaLog_l3", v] + C
        hac, plain, d = lg.estimar(p, xs)
        out.append({"bloco": "B. Definição da expectativa", "especificação": f"L2 + {rot}", "coef Esp": hac.params[v], "p Esp": hac.pvalues[v],
                    "soma ΔMeta 0-3": hac.params["MetaLog_l3"], "p soma ΔMeta": hac.pvalues["MetaLog_l3"], "n": int(hac.nobs)})

    # ---------- (C) colinearidade ----------
    xs = pol1 + ["Esp_d6m"] + C
    d = p.dropna(subset=xs + ["log_N_MA"]).loc["2006-01":mqo.FIM]
    X = sm.add_constant(d[xs])
    vif = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
    pd.Series(vif).to_csv(SAIDA / "C_vif_F1.csv")
    print("C. VIF F1:", {k: round(v, 2) for k, v in vif.items()})
    xs2 = ["MetaLog_l3", "Esp_d6m"] + C
    d2 = p.dropna(subset=xs2 + ["log_N_MA"]).loc["2006-01":mqo.FIM]
    X2 = sm.add_constant(d2[xs2])
    vif2 = {c: variance_inflation_factor(X2.values, i) for i, c in enumerate(X2.columns) if c != "const"}
    print("C. VIF F2:", {k: round(v, 2) for k, v in vif2.items()})
    pd.Series(vif2).to_csv(SAIDA / "C_vif_F2.csv")

    # ---------- (D) endogeneidade: avanço e feedback ----------
    for k in (1, 2, 3):
        p[f"Esp_d6m_lead{k}"] = p["Esp_d6m"].shift(-k)
    xs = ["log_N_MA_lag1", "Esp_d6m"] + [f"Esp_d6m_lead{k}" for k in (1, 2, 3)] + C
    hac, plain, d = lg.estimar(p, xs)
    out.append({"bloco": "D. Endogeneidade", "especificação": "Teste de avanço: Esp_d6m em t+1..t+3 (conjunto)",
                "coef Esp": lg.soma(hac, [f"Esp_d6m_lead{k}" for k in (1, 2, 3)])[0], "p Esp": lg.soma(hac, [f"Esp_d6m_lead{k}" for k in (1, 2, 3)])[2],
                "soma ΔMeta 0-3": np.nan, "p soma ΔMeta": np.nan, "n": int(hac.nobs)})
    # M&A passado prevê a expectativa?
    for k in range(1, 4):
        p[f"logNMA_l{k}"] = p["log_N_MA"].shift(k)
    xs = ["Esp_d6m_l1"] + [f"logNMA_l{k}" for k in (1, 2, 3)] + C
    p["Esp_d6m_l1"] = p["Esp_d6m"].shift(1)
    dd = p.loc["2006-01":mqo.FIM].dropna(subset=["Esp_d6m"] + xs)
    hac, _ = mqo.ols_hac(dd["Esp_d6m"], dd[xs])
    w = hac.t_test(" + ".join([f"logNMA_l{k}" for k in (1, 2, 3)]) + " = 0")
    out.append({"bloco": "D. Endogeneidade", "especificação": "Reverso: ln(M&A) em t-1..t-3 explica Esp_d6m (soma)",
                "coef Esp": float(np.squeeze(w.effect)), "p Esp": float(np.squeeze(w.pvalue)),
                "soma ΔMeta 0-3": np.nan, "p soma ΔMeta": np.nan, "n": int(hac.nobs)})

    # ---------- (E) estabilidade por período ----------
    pers = {"2006-2014": ("2006-01", "2014-12"), "2015-2024": ("2015-01", "2024-12"), "Sem 2020-2022": None, "Completo": ("2006-01", "2024-12")}
    for rot, per in pers.items():
        for nome, xs, foco in (("F3 só expectativa", ["log_N_MA_lag1", "Esp_d6m"] + C, ["Esp_d6m"]),
                               ("F1 L1+Esp", ["log_N_MA_lag1"] + pol1 + ["Esp_d6m"] + C, pol1),
                               ("F2 L2+Esp", ["log_N_MA_lag1", "MetaLog_l3", "Esp_d6m"] + C, ["MetaLog_l3"])):
            if per is None:
                d = p.loc["2006-01":mqo.FIM].dropna(subset=["log_N_MA"] + xs)
                d = d[~d.index.year.isin([2020, 2021, 2022])]
            else:
                d = p.loc[per[0]:per[1]].dropna(subset=["log_N_MA"] + xs)
            hac, _ = mqo.ols_hac(d["log_N_MA"], d[xs])
            b, ep, pv = lg.soma(hac, foco) if len(foco) > 1 else (hac.params[foco[0]], hac.bse[foco[0]], hac.pvalues[foco[0]])
            out.append({"bloco": "E. Estabilidade", "especificação": f"{nome} — {rot}", "coef Esp": hac.params["Esp_d6m"], "p Esp": hac.pvalues["Esp_d6m"],
                        "soma ΔMeta 0-3": b, "p soma ΔMeta": pv, "n": int(hac.nobs)})

    # ---------- (F) controle por ciclo: ln(M&A) detrended ----------
    xs = ["log_N_MA_lag1", "Esp_d6m", "tendencia"] + C
    hac, plain, d = lg.estimar(p, xs)
    out.append({"bloco": "F. Ciclo", "especificação": "F3 + tendência linear", "coef Esp": hac.params["Esp_d6m"], "p Esp": hac.pvalues["Esp_d6m"],
                "soma ΔMeta 0-3": np.nan, "p soma ΔMeta": np.nan, "n": int(hac.nobs)})
    xs = ["log_N_MA_lag1", "Esp_d6m", "IPCA_log"] + C[:3]
    xs = list(dict.fromkeys(xs))
    hac, plain, d = lg.estimar(p, xs)
    out.append({"bloco": "F. Ciclo", "especificação": "F3 + IPCA_log", "coef Esp": hac.params["Esp_d6m"], "p Esp": hac.pvalues["Esp_d6m"],
                "soma ΔMeta 0-3": np.nan, "p soma ΔMeta": np.nan, "n": int(hac.nobs)})

    # ---------- (G) expectativa em NÍVEL (postura esperada da política) ----------
    linhas = []
    for v in ["Meta_log", "E_prox_log", "E_6m_log", "E_12m_log"]:
        for k in (0, 1, 3, 6):
            p[f"{v}_l{k}"] = p[v].shift(k)
            hac, plain, d = lg.estimar(p, ["log_N_MA_lag1", f"{v}_l{k}"] + C)
            linhas.append({"nível usado": v, "k": k, "coef": hac.params[f"{v}_l{k}"], "ep": hac.bse[f"{v}_l{k}"],
                           "p": hac.pvalues[f"{v}_l{k}"], "R2_aj": hac.rsquared_adj, "n": int(hac.nobs)})
    g = pd.DataFrame(linhas)
    g.to_csv(SAIDA / "niveis_expectativa.csv", index=False)
    print("G. Expectativa em nível:\n", g.round(4).to_string())

    res = pd.DataFrame(out)
    res.to_csv(SAIDA / "verificacao_expectativa.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(res.round(4).to_string())

if __name__ == "__main__":
    main()
