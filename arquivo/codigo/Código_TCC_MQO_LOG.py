"""
MQO mensal com TODAS as variáveis em log + expectativa da Selic (Boletim Focus)
===============================================================================

Reaproveita dados, filtros e funções de Código_TCC_MQO_SGS.py (mesmo painel, mesma
amostra, mesmo HAC) e acrescenta duas coisas pedidas pelo aluno:

1) TRANSFORMAÇÃO LOGARÍTMICA DE TODAS AS VARIÁVEIS (exceto a dummy Covid)
   - Contagem de M&A ................ ln(N_MA)                            (já era)
   - Taxas (Selic Meta, IPCA) ....... 100 · ln(1 + taxa/100)               "taxa contínua"
     (o "−1" só entra na volta: taxa = exp(x/100) − 1; dentro do log não se subtrai)
   - Índices/preços em nível ........ ln(nível); a variação vira 100·Δln(nível)
       Ibovespa  -> Ibov_Ret = 100·Δln(Ibov)      (já era)
       IBC-Br    -> dIBC_log  = 100·Δln(IBC-Br)   (antes: pct_change·100)
       Câmbio    -> dCambio_log = 100·Δln(câmbio) (antes: pct_change·100)
       IIE-Br    -> dIIE_log  = 100·Δln(IIE-Br)   (antes: diferença simples)
   - ΔMeta_log = Δ[100·ln(1 + Meta/100)]  (variação da taxa contínua, em "pontos log")
   Variáveis que podem ser zero/negativas por construção (variações, ΔMeta, dummy) NÃO
   entram em log: já são diferenças de logs, que é a forma correta.

2) EXPECTATIVA DA SELIC (Boletim Focus/BCB, mediana por reunião do Copom)
   Para cada fim de mês t, usa a última pesquisa do mês e, entre as reuniões ainda não
   realizadas, extrai a Selic esperada para:
     E_prox  = próxima reunião                       E_6m = reunião mais próxima de t+6 meses
     E_12m   = reunião mais próxima de t+12 meses
   Variáveis derivadas (todas em pontos log):
     Esp_d6m   = ln-taxa(E_6m)  − ln-taxa(Meta_t)    mudança de Selic ESPERADA nos próx. 6 meses
     Esp_d12m  = ln-taxa(E_12m) − ln-taxa(Meta_t)
     Esp_rev6m = variação mensal de ln-taxa(E_6m)    "notícia": revisão da trajetória esperada
   A data da reunião é aproximada pela última data do Focus daquele rótulo (R#/ano).

Modelos (amostra comum; HAC Newey-West 6):
   L1  ln(M&A) ~ ln(M&A)_{t-1} + ΔMeta_log (k=0..3) + controles          [= M1 em log]
   L2  ln(M&A) ~ ln(M&A)_{t-1} + Meta_log_{t-3}        + controles          [= M2 em log]
   F1  L1 + Esp_d6m_t      F2  L2 + Esp_d6m_t     (controle de expectativa)
   F3  ln(M&A) ~ ln(M&A)_{t-1} + Esp_d6m_t + controles                      (só expectativa)
   F4  L1 + Esp_rev6m (k=0..3)                                             (notícia sobre a Selic)
   + comparação ORIGINAL x LOG no mesmo coeficiente, grade k=0..12 para ΔMeta_log,
     grade para as variáveis de expectativa, ADF/KPSS das novas séries e diagnósticos.

Rodar:  python Código_TCC_MQO_LOG.py   (precisa de Deals_Results_TCC_FactSet_Att.xlsx e de
dados_focus/focus_selic_mediana.csv, gerado por Código_TCC_VAR_Surpresa.py). Saída em
./resultados_mqo_log
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_mqo_log"
SAIDA.mkdir(exist_ok=True)
CACHE_FOCUS = BASE / "dados_focus" / "focus_selic_mediana.csv"

spec = importlib.util.spec_from_file_location("mqo", BASE / "Código_TCC_MQO_SGS.py")
mqo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mqo)

L100 = lambda taxa: 100 * np.log1p(taxa / 100.0)          # taxa em % -> log-taxa (×100)
FOCUS_INFO: dict = {}


# ----------------------------------------------------------------------------
# 1. Focus -> variáveis mensais
# ----------------------------------------------------------------------------
def expectativas_focus() -> pd.DataFrame:
    if not CACHE_FOCUS.exists():
        raise SystemExit(f"Falta {CACHE_FOCUS}. Gere com Código_TCC_VAR_Surpresa.py (BAIXAR_FOCUS=True).")
    f = pd.read_csv(CACHE_FOCUS, parse_dates=["Data"])
    data_reuniao = f.groupby("Reuniao")["Data"].max()            # proxy da data da reunião
    f["data_reuniao"] = f["Reuniao"].map(data_reuniao)
    FOCUS_INFO["linhas"] = len(f)
    FOCUS_INFO["periodo"] = f"{f['Data'].min():%Y-%m-%d} a {f['Data'].max():%Y-%m-%d}"
    FOCUS_INFO["reunioes"] = f["Reuniao"].nunique()

    out = {}
    datas = np.sort(f["Data"].unique())
    for per in pd.period_range("2004-12", mqo.FIM, freq="M"):
        fim_mes = per.to_timestamp(how="end")
        ok = datas[datas <= np.datetime64(fim_mes)]
        if len(ok) == 0:
            continue
        s = pd.Timestamp(ok[-1])
        if (fim_mes - s).days > 7:                            # sem pesquisa recente
            continue
        d = f[f["Data"] == s].dropna(subset=["Mediana"])
        if d.empty:
            continue
        d = d.assign(dist=(d["data_reuniao"] - s).dt.days).sort_values("dist")
        reg = {"data_pesquisa": s, "E_prox": d.iloc[0]["Mediana"]}
        for nome, dias in (("E_6m", 182), ("E_12m", 365)):
            j = (d["dist"] - dias).abs().idxmin()
            if abs(d.loc[j, "dist"] - dias) <= 50:
                reg[nome] = d.loc[j, "Mediana"]
        out[per] = reg
    return pd.DataFrame.from_dict(out, orient="index")


def construir(painel: pd.DataFrame) -> pd.DataFrame:
    p = painel.copy()
    # --- tudo em log ---
    p["Meta_log"] = L100(p["Meta"])
    p["dMeta_log"] = p["Meta_log"].diff()
    for k in range(mqo.MAX_LAG + 1):
        p[f"dMetaLog_l{k}"] = p["dMeta_log"].shift(k)
        p[f"MetaLog_l{k}"] = p["Meta_log"].shift(k)
    p["IPCA_log"] = L100(p["IPCA"])
    p["dIBC_log"] = 100 * np.log(p["IBC_Br"]).diff()
    if "Cambio" in p:
        p["dCambio_log"] = 100 * np.log(p["Cambio"]).diff()
    p["dIIE_log"] = 100 * np.log(p["IIE_Br"]).diff()
    # --- Focus ---
    foc = expectativas_focus()
    p = p.join(foc[[c for c in ["E_prox", "E_6m", "E_12m"] if c in foc]], how="left")
    for c in ["E_prox", "E_6m", "E_12m"]:
        if c in p:
            p[c + "_log"] = L100(p[c])
    p["Esp_dprox"] = p["E_prox_log"] - p["Meta_log"]
    p["Esp_d6m"] = p["E_6m_log"] - p["Meta_log"]
    p["Esp_d12m"] = p["E_12m_log"] - p["Meta_log"]
    p["Esp_rev6m"] = p["E_6m_log"].diff()
    for k in range(mqo.MAX_LAG + 1):
        p[f"Esp_d6m_l{k}"] = p["Esp_d6m"].shift(k)
        p[f"Esp_rev6m_l{k}"] = p["Esp_rev6m"].shift(k)
    return p


CTRL_LOG = ["Ibov_Ret", "dIBC_log", "Covid"]
CTRL_ROB_LOG = ["Ibov_Ret", "dIBC_log", "Covid", "IPCA_log", "dCambio_log", "dIIE_log"]


# ----------------------------------------------------------------------------
# 2. estimação
# ----------------------------------------------------------------------------
INI_PADRAO = "2006-01"   # mesma amostra do diagnóstico (n=228): defasagens de 0 a 12 exigem 2006-01


def estimar(p, xs, ini="padrao", fim=mqo.FIM):
    cols = ["log_N_MA"] + xs
    ini = INI_PADRAO if ini == "padrao" else ini
    d = mqo.recorte(p, ini, fim).dropna(subset=cols)
    hac, plain = mqo.ols_hac(d["log_N_MA"], d[xs])
    return hac, plain, d


def soma(hac, vars_):
    t = hac.t_test(" + ".join(vars_) + " = 0")
    return float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))


def linha_modelo(nome, hac, plain, d, foco, soma_vars=None, extra=None):
    if soma_vars:
        b, ep, pv = soma(hac, soma_vars)
        rot = "soma " + "+".join(v.split("_l")[-1] if "_l" in v else v for v in soma_vars)
    else:
        b, ep, pv = float(hac.params[foco]), float(hac.bse[foco]), float(hac.pvalues[foco])
        rot = foco
    bg = acorr_breusch_godfrey(plain, nlags=6)[1]
    jb = jarque_bera(plain.resid)[1]
    bp = het_breuschpagan(plain.resid, plain.model.exog)[1]
    r = {"modelo": nome, "coeficiente-chave": rot, "coef": b, "ep_HAC": ep, "p_valor": pv, "n": int(hac.nobs),
         "R2_aj": hac.rsquared_adj, "amostra": f"{d.index.min()} a {d.index.max()}",
         "BG(6) p": bg, "Jarque-Bera p": jb, "Breusch-Pagan p": bp}
    if extra:
        r.update(extra)
    return r


def coef_linha(hac, v):
    return {"coef": hac.params[v], "ep_HAC": hac.bse[v], "p_valor": hac.pvalues[v]}


# ----------------------------------------------------------------------------
# 3. blocos de resultado
# ----------------------------------------------------------------------------
def comparar_original_log(p):
    """Mesmos modelos com as variáveis ORIGINAIS (pontos/%) e em LOG, na MESMA amostra."""
    comuns = ["log_N_MA", "log_N_MA_lag1", "dMeta_l0", "dMeta_l1", "dMeta_l2", "dMeta_l3", "Meta_l3",
              "Ibov_Ret", "dIBC_Br", "Covid", "dMetaLog_l0", "dMetaLog_l1", "dMetaLog_l2", "dMetaLog_l3",
              "MetaLog_l3", "dIBC_log"]
    linhas = []
    sets = {
        "M1 original (ΔMeta em p.p., ΔIBC em %)": (["log_N_MA_lag1"] + [f"dMeta_l{k}" for k in range(4)] + mqo.CTRL_BASE, [f"dMeta_l{k}" for k in range(4)]),
        "L1 em log (ΔMeta_log, ΔIBC_log)": (["log_N_MA_lag1"] + [f"dMetaLog_l{k}" for k in range(4)] + CTRL_LOG, [f"dMetaLog_l{k}" for k in range(4)]),
        "M2 original (Meta em %, k=3)": (["log_N_MA_lag1", "Meta_l3"] + mqo.CTRL_BASE, ["Meta_l3"]),
        "L2 em log (Meta_log, k=3)": (["log_N_MA_lag1", "MetaLog_l3"] + CTRL_LOG, ["MetaLog_l3"]),
    }
    for rot, ini in (("2006-01+ (padrão)", "padrao"), ("2005-04+ (amostra máxima)", None)):
        for nome, (xs, pol) in sets.items():
            hac, plain, d = estimar(p, xs, ini)
            linhas.append(linha_modelo(nome, hac, plain, d, pol[0], soma_vars=pol if len(pol) > 1 else None, extra={"amostra-base": rot}))
    return pd.DataFrame(linhas)


def grade_log(p):
    linhas = []
    for k in range(mqo.MAX_LAG + 1):
        xs = ["log_N_MA_lag1", f"dMetaLog_l{k}"] + CTRL_LOG
        hac, plain, d = estimar(p, xs)
        linhas.append({"k_meses": k, **coef_linha(hac, f"dMetaLog_l{k}"), "n": int(hac.nobs)})
    g = pd.DataFrame(linhas)
    from statsmodels.stats.multitest import multipletests
    g["p_Holm"] = multipletests(g["p_valor"], method="holm")[1]
    return g


def grade_expectativa(p):
    linhas = []
    for nome, pref in (("Esp_d6m (mudança esperada 6m)", "Esp_d6m_l"), ("Esp_rev6m (revisão da trajetória)", "Esp_rev6m_l")):
        for k in range(0, 7):
            xs = ["log_N_MA_lag1", f"{pref}{k}"] + CTRL_LOG
            hac, plain, d = estimar(p, xs)
            linhas.append({"variável": nome, "k_meses": k, **coef_linha(hac, f"{pref}{k}"), "n": int(hac.nobs)})
    return pd.DataFrame(linhas)


def modelos_focus(p):
    linhas, coefs = [], {}
    pol1 = [f"dMetaLog_l{k}" for k in range(4)]
    spec_ = {
        "L1  ΔMeta_log (0-3) + controles": (["log_N_MA_lag1"] + pol1 + CTRL_LOG, pol1),
        "L2  Meta_log (k=3) + controles": (["log_N_MA_lag1", "MetaLog_l3"] + CTRL_LOG, ["MetaLog_l3"]),
        "F1  L1 + Esp_d6m (expectativa de mudança)": (["log_N_MA_lag1"] + pol1 + ["Esp_d6m"] + CTRL_LOG, pol1),
        "F2  L2 + Esp_d6m": (["log_N_MA_lag1", "MetaLog_l3", "Esp_d6m"] + CTRL_LOG, ["MetaLog_l3"]),
        "F3  só expectativa: Esp_d6m + controles": (["log_N_MA_lag1", "Esp_d6m"] + CTRL_LOG, ["Esp_d6m"]),
        "F4  L1 + revisão Esp_rev6m (0-3)": (["log_N_MA_lag1"] + pol1 + [f"Esp_rev6m_l{k}" for k in range(4)] + CTRL_LOG, pol1),
        "F5  F1 + Esp_d12m": (["log_N_MA_lag1"] + pol1 + ["Esp_d6m", "Esp_d12m"] + CTRL_LOG, pol1),
    }
    # amostra comum a todos
    ini = "padrao"
    for nome, (xs, pol) in spec_.items():
        hac, plain, d = estimar(p, xs, ini)
        extra = {}
        if "Esp_d6m" in xs:
            extra = {"coef Esp_d6m": hac.params["Esp_d6m"], "p Esp_d6m": hac.pvalues["Esp_d6m"]}
        if "Esp_d12m" in xs:
            extra.update({"coef Esp_d12m": hac.params["Esp_d12m"], "p Esp_d12m": hac.pvalues["Esp_d12m"]})
        if any(c.startswith("Esp_rev6m") for c in xs):
            rv = [f"Esp_rev6m_l{k}" for k in range(4)]
            b, ep, pv = soma(hac, rv)
            extra.update({"soma Esp_rev6m (0-3)": b, "p Esp_rev6m (0-3)": pv})
        linhas.append(linha_modelo(nome, hac, plain, d, pol[0], soma_vars=pol if len(pol) > 1 else None, extra=extra))
        coefs[nome] = hac.summary2().tables[1]
    return pd.DataFrame(linhas), coefs, ini


def raiz_unitaria(p):
    cand = {"Meta_log (nível)": "Meta_log", "ΔMeta_log": "dMeta_log", "IPCA_log": "IPCA_log", "Ibov_Ret": "Ibov_Ret",
            "dIBC_log": "dIBC_log", "dCambio_log": "dCambio_log", "dIIE_log": "dIIE_log", "ln(N_MA)": "log_N_MA",
            "E_6m_log (nível)": "E_6m_log", "Esp_d6m": "Esp_d6m", "Esp_d12m": "Esp_d12m", "Esp_rev6m": "Esp_rev6m"}
    linhas = []
    for nome, c in cand.items():
        if c not in p:
            continue
        s = p[c].dropna()
        s = s.loc[mqo.INICIO:mqo.FIM]
        if len(s) < 40:
            continue
        r = {"Variável": nome, "n": len(s)}
        for tag, reg in (("c", "c"), ("c+tend.", "ct")):
            a = adfuller(s, regression=reg, autolag="AIC")
            r[f"ADF {tag}: estat."], r[f"ADF {tag}: p"] = a[0], a[1]
        for tag, reg in (("c", "c"), ("c+tend.", "ct")):
            k = kpss(s, regression=reg, nlags="auto")
            r[f"KPSS {tag}: estat."], r[f"KPSS {tag}: p (trunc.)"] = k[0], k[1]
        adf_ok = r["ADF c: p"] < 0.05
        kp_ok = r["KPSS c: p (trunc.)"] > 0.05
        r["Conclusão"] = ("estacionária (ADF e KPSS concordam)" if adf_ok and kp_ok else
                          "raiz unitária (ADF e KPSS concordam)" if not adf_ok and not kp_ok else "inconclusivo (ADF e KPSS discordam)")
        linhas.append(r)
    return pd.DataFrame(linhas)


def graficos(p, g_log, g_orig, g_exp):
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.plot(g_orig["k_meses"], g_orig["coef"], "o-", color="#7f7f7f", label="ΔMeta original (p.p.)")
    ax.plot(g_log["k_meses"], g_log["coef"], "s-", color="#1f3864", label="ΔMeta_log (pontos log)")
    ax.fill_between(g_log["k_meses"], g_log["coef"] - 1.96 * g_log["ep_HAC"], g_log["coef"] + 1.96 * g_log["ep_HAC"], color="#1f3864", alpha=0.15)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Defasagem k (meses)")
    ax.set_ylabel("Coeficiente sobre ln(M&A)")
    ax.set_title("ΔMeta Selic: original x em log (IC 95% HAC do log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "grade_dMeta_original_vs_log.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.plot(p.index.to_timestamp(), p["Meta"], color="#1f3864", label="Meta Selic (efetiva, fim do mês)")
    if "E_6m" in p:
        ax.plot(p.index.to_timestamp(), p["E_6m"], color="#d62728", lw=1.2, label="Selic esperada p/ ~6 meses à frente (Focus)")
    ax.set_ylabel("% a.a.")
    ax.set_title("Meta Selic x expectativa do mercado (Focus) para ~6 meses à frente")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "selic_vs_expectativa_6m.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    for nome, gr in g_exp.groupby("variável"):
        ax.errorbar(gr["k_meses"], gr["coef"], yerr=1.96 * gr["ep_HAC"], marker="o", capsize=3, label=nome)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Defasagem k (meses)")
    ax.set_ylabel("Coeficiente sobre ln(M&A)")
    ax.set_title("Variáveis de expectativa (Focus) — IC 95% HAC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "grade_expectativa.png")
    plt.close(fig)


def main():
    neg = mqo.aplicar_filtros(mqo.carregar_negocios())
    painel = mqo.montar_painel(neg)
    p = construir(painel)
    pd.DataFrame(mqo.FONTES).T.to_csv(SAIDA / "fontes_dados.csv")
    print(f"Focus: {FOCUS_INFO}")
    print("Cobertura mensal das variáveis de expectativa:",
          {c: int(p.loc[mqo.INICIO:mqo.FIM, c].notna().sum()) for c in ["E_prox", "E_6m", "E_12m"] if c in p})

    comp = comparar_original_log(p)
    comp.to_csv(SAIDA / "comparacao_original_vs_log.csv", index=False)
    print("\n== ORIGINAL x LOG (mesma amostra) ==")
    print(comp[["modelo", "coef", "ep_HAC", "p_valor", "n", "amostra", "BG(6) p", "Jarque-Bera p", "Breusch-Pagan p"]].round(4).to_string())

    g_log = grade_log(p)
    g_log.to_csv(SAIDA / "grade_dMeta_log.csv", index=False)
    g_orig = pd.DataFrame([{"k_meses": k, **coef_linha(estimar(p, ["log_N_MA_lag1", f"dMeta_l{k}"] + mqo.CTRL_BASE)[0], f"dMeta_l{k}")} for k in range(mqo.MAX_LAG + 1)])
    g_orig.to_csv(SAIDA / "grade_dMeta_original.csv", index=False)
    print("\n== GRADE ΔMeta_log, k=0..12 ==")
    print(g_log.round(4).to_string())

    gexp = grade_expectativa(p)
    gexp.to_csv(SAIDA / "grade_expectativa.csv", index=False)
    print("\n== GRADE expectativa ==")
    print(gexp.round(4).to_string())

    mf, coefs, ini = modelos_focus(p)
    mf.to_csv(SAIDA / "modelos_focus.csv", index=False)
    for nome, tab in coefs.items():
        tab.to_csv(SAIDA / f"coef_{nome.split()[0]}.csv")
    print("\n== MODELOS COM EXPECTATIVA (amostra padrão 2006-01+) ==")
    print(mf.round(4).to_string())

    cc = p.loc["2006-01":mqo.FIM, ["Esp_d6m", "Meta_log", "dMeta_log", "E_6m_log", "log_N_MA"]].corr()
    cc.to_csv(SAIDA / "correlacoes_expectativa.csv")
    print("\nCorrelações:\n", cc.round(3).to_string())

    ru = raiz_unitaria(p)
    ru.to_csv(SAIDA / "raiz_unitaria_variaveis_log.csv", index=False)
    print("\n== RAIZ UNITÁRIA ==")
    print(ru[["Variável", "n", "ADF c: p", "ADF c+tend.: p", "KPSS c: p (trunc.)", "KPSS c+tend.: p (trunc.)", "Conclusão"]].round(4).to_string())

    p.loc[mqo.INICIO:mqo.FIM, ["N_MA", "log_N_MA", "Meta", "Meta_log", "dMeta_log", "IPCA", "IPCA_log", "E_prox", "E_6m", "E_12m",
                               "Esp_dprox", "Esp_d6m", "Esp_d12m", "Esp_rev6m", "Ibov_Ret", "dIBC_log", "dIIE_log"]].to_csv(SAIDA / "painel_log_focus.csv")
    graficos(p, g_log, g_orig, gexp)
    print("\nPronto. Saída em", SAIDA)


if __name__ == "__main__":
    main()
