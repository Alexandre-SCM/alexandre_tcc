"""
MODELO FINAL DO TCC — MQO mensal em log, índice de rigor (Stringency), dummies de mês
====================================================================================

Especificação principal (H1 — política monetária em NÍVEL):

  ln(M&A)_t = c + a·ln(M&A)_{t-1} + g_k·Meta_log_{t-k} + b1·Ibov_Ret_t + b2·ΔIBC_log_t
              + b3·Stringency_t + Σ_{m=2..12} d_m·Mês_m + e_t          k ∈ {1, 3, 6, 9, 12}

  Meta_log = 100·ln(1 + Meta Selic/100);  Ibov_Ret = 100·Δln(Ibovespa);  ΔIBC_log = 100·Δln(IBC-Br);
  Stringency = índice de rigor da Oxford (OxCGRT, Brasil, média mensal/100; 0 antes de 2020);
  erros-padrão HAC (Newey-West, 6 defasagens). Sem a expectativa da Selic do Focus (decisão do aluno).

Robustez:
  R1  variação da Meta (ΔMeta_log), efeitos pontuais k=1,3,6 e acumulados 0-3, 0-6
  R2  controles extras (IPCA_log, ΔIIE_log, ΔCâmbio se disponível)
  R3  dummy Covid no lugar do Stringency
  R4  sem dummies de mês (modelo anterior)
  R5  subamostras (2015-2024; sem 2020-2022)
  R6  estimadores alternativos no k=6 (HC3, Poisson+HAC, Huber, mediana)
  R7  previsão fora da amostra (janela expansiva), com e sem a política

Diagnóstico de cada modelo: Breusch-Godfrey(6), Jarque-Bera, Breusch-Pagan, RESET, F das dummies de mês.

Rodar: python Código_TCC_MQO_FINAL.py   (precisa de Código_TCC_MQO_LOG.py, Código_TCC_MQO_STRINGENCY.py
e Código_TCC_MQO_SGS.py na mesma pasta, além de dados_covid/stringency_brasil_mensal.csv ou
Stringency_index.csv). Saída: ./resultados_mqo_final
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
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_mqo_final"
SAIDA.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("st", BASE / "Código_TCC_MQO_STRINGENCY.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)
lg, mqo = st.lg, st.mqo

LAGS = (1, 3, 6, 9, 12)
CTRL = ["Ibov_Ret", "dIBC_log", "Str"]
MESES = [f"M{m}" for m in range(2, 13)]
AR1 = ["log_N_MA_lag1"]


def preparar():
    neg = mqo.aplicar_filtros(mqo.carregar_negocios())
    p = st.juntar(lg.construir(mqo.montar_painel(neg)))
    for m in range(2, 13):
        p[f"M{m}"] = (p.index.month == m).astype(int)
    for k in range(0, 13):
        p[f"MetaLog_l{k}"] = p["Meta_log"].shift(k)
    return p


def ajuste(p, xs, ini="padrao", fim=mqo.FIM, excluir_anos=None):
    ini = lg.INI_PADRAO if ini == "padrao" else ini
    d = mqo.recorte(p, ini, fim).dropna(subset=["log_N_MA"] + xs)
    if excluir_anos:
        d = d[~d.index.year.isin(excluir_anos)]
    xs = [x for x in xs if d[x].nunique() > 1]
    X = sm.add_constant(d[xs])
    plain = sm.OLS(d["log_N_MA"], X).fit()
    hac = sm.OLS(d["log_N_MA"], X).fit(cov_type="HAC", cov_kwds={"maxlags": mqo.NW_LAGS, "use_correction": True}, use_t=True)
    return hac, plain, d, xs


def diag(plain, hac, meses_in):
    r = {"R2_aj": plain.rsquared_adj, "AIC": plain.aic, "BIC": plain.bic,
         "BG(6) p": acorr_breusch_godfrey(plain, nlags=6)[1], "JB p": jarque_bera(plain.resid)[1],
         "BP p": het_breuschpagan(plain.resid, plain.model.exog)[1]}
    try:
        r["RESET p"] = float(linear_reset(plain, power=2, use_f=True).pvalue)
    except Exception:  # noqa: BLE001
        r["RESET p"] = np.nan
    if meses_in:
        r["F meses p (HAC)"] = float(hac.wald_test(", ".join(f"{m} = 0" for m in meses_in), use_f=True, scalar=True).pvalue)
    return r


def chave(hac, pol):
    if len(pol) > 1:
        t = hac.t_test(" + ".join(pol) + " = 0")
        b, ep, pv = float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))
        ic = t.conf_int()
        lo, hi = float(np.squeeze(ic)[0]), float(np.squeeze(ic)[1])
    else:
        b, ep, pv = float(hac.params[pol[0]]), float(hac.bse[pol[0]]), float(hac.pvalues[pol[0]])
        lo, hi = [float(x) for x in hac.conf_int().loc[pol[0]]]
    return b, ep, pv, lo, hi


def linha(nome, p, pol, extra_x, meses=True, **kw):
    xs = AR1 + pol + extra_x + (MESES if meses else [])
    hac, plain, d, xs_ok = ajuste(p, xs, **kw)
    meses_in = [m for m in MESES if m in xs_ok]
    b, ep, pv, lo, hi = chave(hac, pol)
    r = {"modelo": nome, "coef": b, "ep_HAC": ep, "p": pv, "ic95_inf": lo, "ic95_sup": hi, "n": int(hac.nobs),
         "amostra": f"{d.index.min()} a {d.index.max()}"}
    r.update(diag(plain, hac, meses_in))
    return r, hac


def main():
    p = preparar()
    pd.set_option("display.width", 230, "display.max_columns", 30)

    # ---------- principal ----------
    prin = []
    hacs = {}
    for k in LAGS:
        r, h = linha(f"Nível Meta_log, k={k}", p, [f"MetaLog_l{k}"], CTRL)
        r["k"] = k
        r["coef Stringency"] = h.params["Str"]
        r["p Stringency"] = h.pvalues["Str"]
        prin.append(r)
        hacs[k] = h
    P = pd.DataFrame(prin)
    P["p_Holm"] = multipletests(P["p"], method="holm")[1]
    P.to_csv(SAIDA / "principal_nivel_por_defasagem.csv", index=False)
    print("== PRINCIPAL: nível da Meta (log), com dummies de mês e Stringency ==")
    print(P[["modelo", "coef", "ep_HAC", "p", "p_Holm", "ic95_inf", "ic95_sup", "n", "R2_aj", "BG(6) p", "JB p", "BP p", "RESET p", "F meses p (HAC)"]].round(4).to_string())

    # tabela completa de coeficientes do modelo de referência k=6
    hacs[6].summary2().tables[1].to_csv(SAIDA / "coeficientes_k6.csv")
    print("\nCoeficientes do modelo de referência (k=6):")
    print(hacs[6].summary2().tables[1].round(4).to_string())

    # ---------- robustez ----------
    R = []
    pol_d = lambda k: [f"dMetaLog_l{k}"]
    for k in (1, 3, 6):
        r, _ = linha(f"R1 ΔMeta_log pontual, k={k}", p, pol_d(k), CTRL)
        R.append(r)
    r, _ = linha("R1 ΔMeta_log acumulado 0-3", p, [f"dMetaLog_l{k}" for k in range(4)], CTRL); R.append(r)
    r, _ = linha("R1 ΔMeta_log acumulado 0-6", p, [f"dMetaLog_l{k}" for k in range(7)], CTRL); R.append(r)
    ex = [c for c in ["IPCA_log", "dIIE_log", "dCambio_log"] if c in p and p[c].notna().sum() > 100]
    r, _ = linha(f"R2 nível k=6 + {', '.join(ex)}", p, ["MetaLog_l6"], CTRL + ex); R.append(r)
    r, _ = linha("R3 nível k=6, dummy Covid no lugar do Stringency", p, ["MetaLog_l6"], ["Ibov_Ret", "dIBC_log", "Covid"]); R.append(r)
    r, _ = linha("R4 nível k=6, sem dummies de mês", p, ["MetaLog_l6"], CTRL, meses=False); R.append(r)
    r, _ = linha("R5 nível k=6, 2015-2024", p, ["MetaLog_l6"], CTRL, ini="2015-01"); R.append(r)
    r, _ = linha("R5 nível k=6, sem 2020-2022", p, ["MetaLog_l6"], CTRL, excluir_anos=[2020, 2021, 2022]); R.append(r)
    r, _ = linha("R5 nível k=6, até 2019", p, ["MetaLog_l6"], CTRL, fim="2019-12"); R.append(r)
    RB = pd.DataFrame(R)
    RB.to_csv(SAIDA / "robustez.csv", index=False)
    print("\n== ROBUSTEZ ==")
    print(RB[["modelo", "coef", "ep_HAC", "p", "n", "R2_aj", "BG(6) p", "JB p", "BP p"]].round(4).to_string())

    # ---------- estimadores alternativos (k=6) ----------
    xs = AR1 + ["MetaLog_l6"] + CTRL + MESES
    d = mqo.recorte(p, lg.INI_PADRAO, mqo.FIM).dropna(subset=["log_N_MA", "N_MA"] + xs)
    X = sm.add_constant(d[xs])
    alt = []

    def add(nome, res, var="MetaLog_l6"):
        ic = res.conf_int().loc[var]
        alt.append({"método": nome, "coef": float(res.params[var]), "ep": float(res.bse[var]), "p": float(res.pvalues[var]),
                    "ic95_inf": float(ic.iloc[0]), "ic95_sup": float(ic.iloc[1])})
    add("MQO + HAC (padrão)", sm.OLS(d["log_N_MA"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True}, use_t=True))
    add("MQO + HC3", sm.OLS(d["log_N_MA"], X).fit(cov_type="HC3"))
    add("Poisson (QML) + HAC, usa a contagem", sm.GLM(d["N_MA"], X, family=sm.families.Poisson()).fit(cov_type="HAC", cov_kwds={"maxlags": 6}))
    add("Regressão robusta (Huber)", sm.RLM(d["log_N_MA"], X, M=sm.robust.norms.HuberT()).fit())
    qr = sm.QuantReg(d["log_N_MA"], X).fit(q=0.5)
    add("Regressão mediana", qr)
    A = pd.DataFrame(alt)
    A.to_csv(SAIDA / "estimadores_alternativos_k6.csv", index=False)
    print("\n== ESTIMADORES ALTERNATIVOS (k=6) ==")
    print(A.round(4).to_string())

    # ---------- previsão fora da amostra ----------
    CT2 = ["Ibov_Ret", "dIBC_log"]
    oos = []
    for nome, xs in (("AR(1)", AR1), ("AR(1) + controles", AR1 + CT2), ("+ dummies de mês", AR1 + CT2 + MESES),
                     ("+ Meta k=6", AR1 + CT2 + ["MetaLog_l6"]), ("+ Meta k=9", AR1 + CT2 + ["MetaLog_l9"]),
                     ("+ dummies de mês + Meta k=6", AR1 + CT2 + MESES + ["MetaLog_l6"]),
                     ("+ dummies de mês + Meta k=9", AR1 + CT2 + MESES + ["MetaLog_l9"])):
        dd = mqo.recorte(p, lg.INI_PADRAO, mqo.FIM).dropna(subset=["log_N_MA"] + xs)
        e = []
        for i in range(108, len(dd)):
            tr, te = dd.iloc[:i], dd.iloc[i:i + 1]
            m = sm.OLS(tr["log_N_MA"], sm.add_constant(tr[xs])).fit()
            e.append(float(te["log_N_MA"].iloc[0] - m.predict(sm.add_constant(te[xs], has_constant="add")).iloc[0]))
        oos.append({"modelo": nome, "RMSE": float(np.sqrt(np.mean(np.square(e)))), "MAE": float(np.mean(np.abs(e))), "n_prev": len(e)})
    O = pd.DataFrame(oos)
    O["RMSE vs AR(1) (%)"] = 100 * (O["RMSE"] / O.loc[0, "RMSE"] - 1)
    O.to_csv(SAIDA / "previsao_fora_da_amostra.csv", index=False)
    print("\n== PREVISÃO FORA DA AMOSTRA ==")
    print(O.round(4).to_string())

    # ---------- gráficos ----------
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    ax.errorbar(P["k"], P["coef"], yerr=[P["coef"] - P["ic95_inf"], P["ic95_sup"] - P["coef"]], fmt="s-", color="#1f3864", capsize=4)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(list(LAGS))
    ax.set_xlabel("Defasagem k (meses)")
    ax.set_ylabel("Coeficiente de Meta_log sobre ln(M&A)")
    ax.set_title("Efeito do nível da Selic sobre M&A, por defasagem (IC 95% HAC)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "coef_nivel_por_defasagem.png")
    plt.close(fig)

    h6 = hacs[6]
    d6 = mqo.recorte(p, lg.INI_PADRAO, mqo.FIM).dropna(subset=["log_N_MA"] + AR1 + ["MetaLog_l6"] + CTRL + MESES)
    res = h6.resid
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), dpi=150)
    ax[0].plot(d6.index.to_timestamp(), d6["log_N_MA"], color="#555555", lw=1.2, label="ln(M&A) observado")
    ax[0].plot(d6.index.to_timestamp(), h6.fittedvalues, color="#1f3864", lw=1.2, label="ajustado (k=6)")
    ax[0].set_title("Modelo final (k=6): observado e ajustado")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)
    ax[1].plot(d6.index.to_timestamp(), res, color="#1f3864", lw=1)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(2 * res.std(), color="#d62728", lw=0.8, ls="--")
    ax[1].axhline(-2 * res.std(), color="#d62728", lw=0.8, ls="--")
    ax[1].set_title("Resíduos")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "ajuste_e_residuos_k6.png")
    plt.close(fig)
    pd.DataFrame({"ajustado": h6.fittedvalues, "residuo": res}).to_csv(SAIDA / "residuos_k6.csv")
    print("\nPronto. Saída em", SAIDA)


if __name__ == "__main__":
    main()
