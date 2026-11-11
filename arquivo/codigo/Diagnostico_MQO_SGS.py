"""
Diagnóstico completo do MQO (Código_TCC_MQO_SGS.py) à luz das hipóteses clássicas
=================================================================================

Reaproveita os dados e as funções de Código_TCC_MQO_SGS.py (mesmas séries do SGS,
mesmos filtros, mesma amostra) e responde, para dois modelos de referência:

  M1  log(M&A)_t = c + a·log(M&A)_{t-1} + b0..b3·ΔMeta_{t..t-3} + Ibov_Ret + ΔIBC-Br + Covid + e
      (efeito acumulado 0-3 da variação da Meta Selic)
  M2  log(M&A)_t = c + a·log(M&A)_{t-1} + g·Meta_{t-3} + Ibov_Ret + ΔIBC-Br + Covid + e
      (nível da Meta Selic, defasagem 3)

DUMMIES DOS MODELOS: apenas Covid (=1 de 2020-03 a 2020-12). Não há dummies de mês
(sazonalidade foi só testada nas versões 2-4) nem dummies de outlier no modelo-base;
as dummies de outlier aparecem apenas na alternativa "MQO + dummies de outlier".

O que sai (pasta ./resultados_diagnostico):
  - residuos_serie_<M>.png           série temporal dos resíduos (±2σ e ±3σ, Covid sombreado)
  - residuos_normalidade_<M>.png     histograma + normal + KDE e gráfico Q-Q
  - residuos_autocorrelacao_<M>.png  FAC, FACP e FAC dos resíduos ao quadrado
  - residuos_vs_ajustados_<M>.png    resíduos x valores ajustados (heterocedasticidade)
  - hipoteses_gauss_markov.csv       as seis hipóteses (TS.1-TS.6) com teste e resultado
  - raiz_unitaria_todas_variaveis.csv ADF (c e c+tendência) e KPSS de cada variável e dos resíduos
  - alternativas_<M>.csv             o mesmo coeficiente estimado por 11 métodos
  - residuos_<M>.csv                 resíduos, ajustados e resíduos padronizados por mês
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
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, acorr_ljungbox, het_arch, het_breuschpagan,
                                          het_white, linear_reset, breaks_cusumolsresid)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_diagnostico"
SAIDA.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("mqo", BASE / "Código_TCC_MQO_SGS.py")
mqo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mqo)

B_BOOT, BLOCO = 2000, 6
RNG = np.random.default_rng(20260707)


# ----------------------------------------------------------------------------
# Modelos de referência
# ----------------------------------------------------------------------------
def dados_modelo(painel, tipo):
    ctrl = mqo.controles_disponiveis(painel, mqo.CTRL_BASE)
    pol = [f"dMeta_l{k}" for k in range(4)] if tipo == "M1" else ["Meta_l3"]
    cols = ["log_N_MA_lag1"] + pol + ctrl
    d = mqo.recorte(painel, mqo.inicio_comum(painel, ctrl), mqo.FIM)
    d = d.dropna(subset=["log_N_MA", "N_MA"] + cols)
    return d, cols, pol


def ajustar(d, cols):
    X = sm.add_constant(d[cols])
    return sm.OLS(d["log_N_MA"], X).fit(), X


def combinacao(res, pol):
    """Estimativa, EP e p-valor de soma(pol)=0 (para M2 é o próprio coeficiente)."""
    expr = " + ".join(pol) + " = 0"
    t = res.t_test(expr)
    ci = np.squeeze(t.conf_int())
    return float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue)), float(ci[0]), float(ci[1])


# ----------------------------------------------------------------------------
# Gráficos dos resíduos
# ----------------------------------------------------------------------------
def graficos(nome, d, plain, titulo):
    r = plain.resid
    z = r / r.std(ddof=1)
    datas = d.index.to_timestamp()
    ci0, ci1 = pd.Period(mqo.COVID_INICIO).to_timestamp(), pd.Period(mqo.COVID_FIM).to_timestamp()

    # 1) série temporal
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(datas, r.values, color="#1f4e79", lw=1.2, marker="o", ms=2.5)
    s = r.std(ddof=1)
    for m, ls, c in ((2, "--", "#b7472a"), (3, ":", "#7a1f10")):
        ax.axhline(m * s, color=c, ls=ls, lw=.9)
        ax.axhline(-m * s, color=c, ls=ls, lw=.9, label=f"±{m}σ" if m == 2 else None)
    ax.axhline(0, color="k", lw=.8)
    ax.axvspan(ci0, ci1, color="gray", alpha=.18, label="Covid (dummy = 1)")
    for dt, v in zip(datas[np.abs(z.values) > 2.5], r.values[np.abs(z.values) > 2.5]):
        ax.annotate(dt.strftime("%m/%Y"), (dt, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_title(f"Resíduos do MQO no tempo — {titulo}")
    ax.set_ylabel("resíduo (log pontos)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(SAIDA / f"residuos_serie_{nome}.png", dpi=200)
    plt.close(fig)

    # 2) normalidade
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(r, bins=24, density=True, color="#9db7d4", edgecolor="white")
    xs = np.linspace(r.min(), r.max(), 300)
    ax[0].plot(xs, stats.norm.pdf(xs, r.mean(), r.std(ddof=1)), color="#b7472a", label="Normal (mesma média e dp)")
    ax[0].plot(xs, stats.gaussian_kde(r)(xs), color="#1f4e79", ls="--", label="densidade estimada (KDE)")
    ax[0].set_title("Distribuição dos resíduos")
    ax[0].legend(fontsize=8)
    sm.qqplot(r, line="s", ax=ax[1], markersize=3)
    ax[1].set_title("Gráfico Q-Q contra a Normal")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(SAIDA / f"residuos_normalidade_{nome}.png", dpi=200)
    plt.close(fig)

    # 3) autocorrelação
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    plot_acf(r, lags=24, ax=ax[0], title="FAC dos resíduos")
    plot_pacf(r, lags=24, ax=ax[1], title="FACP dos resíduos", method="ywm")
    plot_acf(r ** 2, lags=24, ax=ax[2], title="FAC dos resíduos ao quadrado (ARCH)")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(SAIDA / f"residuos_autocorrelacao_{nome}.png", dpi=200)
    plt.close(fig)

    # 4) resíduos x ajustados
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(plain.fittedvalues, r, s=12, color="#1f4e79", alpha=.7)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("valor ajustado de log(M&A)")
    ax.set_ylabel("resíduo")
    ax.set_title(f"Resíduos x ajustados — {titulo}", fontsize=10)
    fig.tight_layout()
    fig.savefig(SAIDA / f"residuos_vs_ajustados_{nome}.png", dpi=200)
    plt.close(fig)

    out = pd.DataFrame({"residuo": r, "ajustado": plain.fittedvalues, "residuo_padronizado": z,
                        "estudentizado_externo": plain.get_influence().resid_studentized_external})
    out.index = d.index
    out.to_csv(SAIDA / f"residuos_{nome}.csv")
    return out


# ----------------------------------------------------------------------------
# As seis hipóteses (Wooldridge TS.1-TS.6) — testes
# ----------------------------------------------------------------------------
def seis_hipoteses(nome, d, cols, pol, plain, X, painel):
    r = plain.resid
    linhas = []

    def add(h, teste, estat, p, leitura):
        linhas.append({"modelo": nome, "hipótese": h, "teste": teste, "estatística": estat, "p-valor": p,
                       "leitura": leitura})

    # TS.1 linearidade nos parâmetros / forma funcional / estabilidade
    f = linear_reset(plain, power=2, use_f=True)
    add("TS.1 Linear nos parâmetros", "RESET (Ramsey, ajustados²)", float(f.fvalue), float(f.pvalue),
        "forma funcional adequada" if f.pvalue > .05 else "indício de não linearidade/variável omitida")
    cs = breaks_cusumolsresid(r.values, ddof=X.shape[1])
    add("TS.1 (estabilidade dos parâmetros)", "CUSUM de resíduos OLS", float(cs[0]), float(cs[1]),
        "parâmetros estáveis" if cs[1] > .05 else "indício de quebra estrutural")

    # TS.2 sem colinearidade perfeita
    vif = max(variance_inflation_factor(X.values, i) for i in range(1, X.shape[1]))
    add("TS.2 Sem colinearidade perfeita", "VIF máximo / nº de condição", float(vif), np.nan,
        f"VIF máx {vif:.2f}; nº de condição {plain.condition_number:.0f} (VIF<5 = sem problema)")

    # TS.3 média condicional zero / exogeneidade estrita: teste de avanço (lead) da variável de política
    if nome == "M1":
        lead = painel["dMeta"].shift(-1).reindex(d.index)
    else:
        lead = painel["Meta"].shift(-1).reindex(d.index)   # Meta em t+1
    dd = d.copy()
    dd["lead"] = lead
    dd = dd.dropna(subset=["lead"])
    Xl = sm.add_constant(dd[cols + ["lead"]])
    hl = sm.OLS(dd["log_N_MA"], Xl).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True}, use_t=True)
    add("TS.3 Exogeneidade (média condicional zero)", "Teste de avanço: coef. da política em t+1 (HAC)",
        float(hl.tvalues["lead"]), float(hl.pvalues["lead"]),
        ("não rejeita exogeneidade estrita em relação à Selic" if hl.pvalues["lead"] > .05 else
         "REJEITA: a Selic futura ajuda a explicar M&A de hoje (feedback / reação do Copom)")
        + ". Obs.: com log(M&A) defasado como regressor, a exogeneidade ESTRITA nunca vale; só a contemporânea.")

    # TS.4 homocedasticidade
    bp = het_breuschpagan(r, plain.model.exog)
    add("TS.4 Homocedasticidade", "Breusch-Pagan", float(bp[0]), float(bp[1]),
        "não rejeita" if bp[1] > .05 else "REJEITA homocedasticidade")
    try:
        w = het_white(r, plain.model.exog)
        add("TS.4 Homocedasticidade", "White", float(w[0]), float(w[1]),
            "não rejeita" if w[1] > .05 else "REJEITA homocedasticidade")
    except Exception:
        pass
    ar = het_arch(r, nlags=6)
    add("TS.4 Homocedasticidade (no tempo)", "ARCH-LM (6 defasagens)", float(ar[0]), float(ar[1]),
        "sem volatilidade agrupada" if ar[1] > .05 else "REJEITA: volatilidade dos erros agrupa no tempo")

    # TS.5 sem autocorrelação serial
    for L in (1, 6, 12):
        bg = acorr_breusch_godfrey(plain, nlags=L)
        add("TS.5 Sem autocorrelação serial", f"Breusch-Godfrey ({L})", float(bg[0]), float(bg[1]),
            "sem autocorrelação" if bg[1] > .05 else "REJEITA ausência de autocorrelação")
    lb = acorr_ljungbox(r, lags=[12], return_df=True)
    add("TS.5 Sem autocorrelação serial", "Ljung-Box (12)", float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0]),
        "sem autocorrelação" if lb["lb_pvalue"].iloc[0] > .05 else "REJEITA ausência de autocorrelação")
    add("TS.5 (aviso)", "Durbin-Watson", float(sm.stats.durbin_watson(r)), np.nan,
        "NÃO usar como teste: inválido com variável dependente defasada entre os regressores")

    # TS.6 normalidade
    jb = jarque_bera(r)
    add("TS.6 Normalidade dos erros", "Jarque-Bera", float(jb[0]), float(jb[1]),
        f"assimetria {jb[2]:.2f}, curtose {jb[3]:.2f} (normal = 0 e 3)" + (" — não rejeita" if jb[1] > .05 else " — REJEITA"))
    sw = stats.shapiro(r)
    add("TS.6 Normalidade dos erros", "Shapiro-Wilk", float(sw[0]), float(sw[1]),
        "não rejeita" if sw[1] > .05 else "REJEITA normalidade")

    # outliers / influência
    infl = plain.get_influence()
    t = infl.resid_studentized_external
    cook = infl.cooks_distance[0]
    add("Observações influentes", "|resíduo estudentizado| > 3 / Cook máx", float((np.abs(t) > 3).sum()), np.nan,
        f"{int((np.abs(t) > 3).sum())} obs. com |t|>3; Cook máx = {cook.max():.3f} em {d.index[int(np.argmax(cook))]}")
    return linhas


# ----------------------------------------------------------------------------
# Raiz unitária de cada variável (e dos resíduos)
# ----------------------------------------------------------------------------
def tabela_adf(painel, residuos):
    vars_ = {
        "log(N_MA)  [dependente]": painel["log_N_MA"], "N_MA (contagem, nível)": painel["N_MA"],
        "Meta Selic (nível)": painel["Meta"], "ΔMeta Selic": painel["dMeta"],
        "Ibov_Ret": painel["Ibov_Ret"], "IBC-Br (nível)": painel["IBC_Br"], "ΔIBC-Br (%)": painel["dIBC_Br"],
        "IPCA": painel["IPCA"], "IIE-Br (nível)": painel["IIE_Br"], "ΔIIE-Br": painel["dIIE_Br"],
    }
    if "dCambio" in painel:
        vars_["ΔCâmbio (%)"] = painel["dCambio"]
    for k, v in residuos.items():
        vars_[f"Resíduo {k}"] = v
    linhas = []
    for nome, s in vars_.items():
        s = s.loc["2005-01":"2024-12"].dropna() if isinstance(s.index, pd.PeriodIndex) else s.dropna()
        if len(s) < 30:
            continue
        a = adfuller(s, regression="c", autolag="AIC")
        b = adfuller(s, regression="ct", autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
        kt = kpss(s, regression="ct", nlags="auto")
        if a[1] < .05 and k[1] >= .05:
            concl = "estacionária (ADF e KPSS concordam)"
        elif a[1] >= .05 and k[1] < .05:
            concl = "raiz unitária / muito persistente (ADF e KPSS concordam)"
        elif a[1] < .05 and k[1] < .05 and b[1] < .05 and kt[1] >= .05:
            concl = "estacionária em torno de tendência (ADF c+tend. e KPSS c+tend. concordam)"
        else:
            concl = "testes discordam / inconclusivo"
        linhas.append({"Variável": nome, "n": len(s), "ADF c: estat.": a[0], "ADF c: p": a[1],
                       "ADF c: valor crítico 5%": a[4]["5%"], "ADF c+tend.: estat.": b[0], "ADF c+tend.: p": b[1],
                       "KPSS c: estat.": k[0], "KPSS c: p (trunc.)": k[1],
                       "KPSS c+tend.: estat.": kt[0], "KPSS c+tend.: p (trunc.)": kt[1], "Conclusão": concl})
    dummy = {"Variável": "Covid (dummy 0/1)", "n": len(painel.loc[mqo.INICIO:mqo.FIM]), "Conclusão": "dummy — ADF não se aplica"}
    linhas.append(dummy)
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Alternativas para heterocedasticidade e não normalidade
# ----------------------------------------------------------------------------
def alternativas(nome, d, cols, pol, plain, X):
    y = d["log_N_MA"]
    linhas = []

    def reg(metodo, est, ep, p, lo, hi, nota):
        linhas.append({"método": metodo, "estimativa": est, "ep": ep, "p-valor": p, "ic95_inf": lo, "ic95_sup": hi,
                       "o que corrige": nota})

    reg("1. MQO, erros-padrão clássicos", *combinacao(plain, pol), "nada (só vale com todas as hipóteses)")
    h3 = sm.OLS(y, X).fit(cov_type="HC3")
    reg("2. MQO + HC3 (White)", *combinacao(h3, pol), "heterocedasticidade")
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True}, use_t=True)
    reg("3. MQO + HAC Newey-West (6)  [padrão do script]", *combinacao(hac, pol), "heterocedasticidade + autocorrelação")

    # FGLS (variância multiplicativa de Harvey)
    e2 = np.log(plain.resid ** 2 + 1e-8)
    aux = sm.OLS(e2, X).fit()
    pesos = 1.0 / np.exp(aux.fittedvalues)
    fg = sm.WLS(y, X, weights=pesos).fit(cov_type="HC3")
    reg("4. FGLS/MQP (variância de Harvey) + HC3", *combinacao(fg, pol), "heterocedasticidade (modela a variância)")

    # GLSAR / Cochrane-Orcutt
    try:
        ga = sm.GLSAR(y, X, rho=1).iterative_fit(maxiter=15)
        reg(f"5. GLSAR (Cochrane-Orcutt, ρ={float(np.squeeze(ga.model.rho)):.2f})", *combinacao(ga, pol), "autocorrelação AR(1)")
    except Exception as e:  # noqa: BLE001
        reg("5. GLSAR", np.nan, np.nan, np.nan, np.nan, np.nan, f"falhou: {e}")

    # dummies de outlier
    t = plain.get_influence().resid_studentized_external
    lim = 3.0
    idx = np.where(np.abs(t) > lim)[0]
    if len(idx) == 0:
        lim, idx = 2.5, np.where(np.abs(t) > 2.5)[0]
    Xo = X.copy()
    for i in idx:
        Xo[f"out_{d.index[i]}"] = (np.arange(len(d)) == i).astype(int)
    ho = sm.OLS(y, Xo).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True}, use_t=True)
    reg(f"6. MQO + {len(idx)} dummies de outlier (|t|>{lim}) + HAC", *combinacao(ho, pol), "outliers / caudas pesadas")

    # robusta (Huber) e mediana
    rl = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    reg("7. Regressão robusta (Huber)", *combinacao(rl, pol), "outliers / não normalidade")
    q = sm.QuantReg(y, X).fit(q=0.5)
    reg("8. Regressão mediana (quantil 0,5)", *combinacao(q, pol), "outliers / não normalidade")

    # Poisson QML com HAC (contagem)
    cnt = sm.GLM(d["N_MA"], X, family=sm.families.Poisson()).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    reg("9. Poisson (QML) + HAC — usa a contagem, sem log", *combinacao(cnt, pol), "natureza de contagem / não normalidade")

    # bootstrap por blocos (pares)
    Xa, ya = X.values, y.values
    n = len(ya)
    w = np.array([1.0 if c in pol else 0.0 for c in X.columns])
    nb = int(np.ceil(n / BLOCO))
    est = np.empty(B_BOOT)
    for b in range(B_BOOT):
        starts = RNG.integers(0, n - BLOCO + 1, nb)
        ids = np.concatenate([np.arange(s, s + BLOCO) for s in starts])[:n]
        beta = np.linalg.lstsq(Xa[ids], ya[ids], rcond=None)[0]
        est[b] = w @ beta
    pt = float(w @ np.linalg.lstsq(Xa, ya, rcond=None)[0])
    p_b = 2 * min((est <= 0).mean(), (est >= 0).mean())
    reg(f"10. Bootstrap em blocos (pares, bloco {BLOCO}, {B_BOOT} réplicas)", pt, float(est.std(ddof=1)), float(max(p_b, 1 / B_BOOT)),
        float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), "não depende de normalidade nem de forma da variância")

    # wild bootstrap (Rademacher) — heterocedasticidade
    res = plain.resid.values
    fit = plain.fittedvalues.values
    est2 = np.empty(B_BOOT)
    for b in range(B_BOOT):
        yb = fit + res * RNG.choice([-1.0, 1.0], n)
        est2[b] = w @ np.linalg.lstsq(Xa, yb, rcond=None)[0]
    # p do wild bootstrap: distribuição centrada na estimativa; compara |est2 - pt| com |pt|
    p_w = float(max((np.abs(est2 - pt) >= abs(pt)).mean(), 1 / B_BOOT))
    reg(f"11. Wild bootstrap (Rademacher, {B_BOOT} réplicas)", pt, float(est2.std(ddof=1)), p_w,
        float(np.percentile(est2, 2.5)), float(np.percentile(est2, 97.5)), "heterocedasticidade de forma desconhecida")
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
def main():
    negocios = mqo.aplicar_filtros(mqo.carregar_negocios())
    painel = mqo.montar_painel(negocios)
    print("Fontes das séries:", {k: v[1][:60] for k, v in mqo.FONTES.items()})

    modelos = {"M1": "M1 — variação da Meta, efeito acumulado 0-3", "M2": "M2 — nível da Meta, defasagem 3"}
    hip, residuos = [], {}
    for nome, titulo in modelos.items():
        print("=" * 72, "\n", titulo)
        d, cols, pol = dados_modelo(painel, nome)
        plain, X = ajustar(d, cols)
        print(f"n = {int(plain.nobs)} | R² ajustado = {plain.rsquared_adj:.3f} | regressores: {list(X.columns)}")
        out = graficos(nome, d, plain, titulo)
        residuos[nome] = plain.resid
        hip += seis_hipoteses(nome, d, cols, pol, plain, X, painel)
        alt = alternativas(nome, d, cols, pol, plain, X)
        alt.to_csv(SAIDA / f"alternativas_{nome}.csv", index=False)
        print(alt.drop(columns="o que corrige").round(4).to_string(index=False))
        top = out["estudentizado_externo"].abs().sort_values(ascending=False).head(5)
        print("Maiores resíduos (|t| estudentizado):", {str(k): round(float(out.loc[k, 'estudentizado_externo']), 2) for k in top.index})

    h = pd.DataFrame(hip)
    h.to_csv(SAIDA / "hipoteses_gauss_markov.csv", index=False)
    print("\n", h[["modelo", "hipótese", "teste", "estatística", "p-valor", "leitura"]].round(4).to_string(index=False))

    adf = tabela_adf(painel, residuos)
    adf.to_csv(SAIDA / "raiz_unitaria_todas_variaveis.csv", index=False)
    print("\n", adf.round(4).to_string(index=False))
    print("\nArquivos em:", SAIDA)


if __name__ == "__main__":
    main()
