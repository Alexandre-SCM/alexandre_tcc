"""
Bateria de melhorias sobre o modelo final (log + Stringency, sem expectativa).
Importa Código_TCC_MQO_STRINGENCY.py. Saída: ./resultados_melhorias
"""
import importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_melhorias"; SAIDA.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("st", BASE / "Código_TCC_MQO_STRINGENCY.py")
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
lg, mqo = st.lg, st.mqo
CT = ["Ibov_Ret", "dIBC_log", "Str"]
POL1 = [f"dMetaLog_l{k}" for k in range(4)]

def fit(p, xs, ini="padrao"):
    d = mqo.recorte(p, lg.INI_PADRAO if ini == "padrao" else ini, mqo.FIM).dropna(subset=["log_N_MA"] + xs)
    X = sm.add_constant(d[xs]); r = sm.OLS(d["log_N_MA"], X).fit()
    h = sm.OLS(d["log_N_MA"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True}, use_t=True)
    return r, h, d

def main():
    neg = mqo.aplicar_filtros(mqo.carregar_negocios())
    p = st.juntar(lg.construir(mqo.montar_painel(neg)))
    p["mes"] = p.index.month
    for m in range(2, 13): p[f"M{m}"] = (p["mes"] == m).astype(int)
    for k in range(2, 5): p[f"log_N_MA_lag{k}"] = p["log_N_MA"].shift(k)
    meses = [f"M{m}" for m in range(2, 13)]
    out = []
    def add(teste, **kw): out.append({"teste": teste, **kw}); print(teste, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()})

    for fam, pol, base in (("L1", POL1, ["log_N_MA_lag1"] + POL1), ("L2", ["MetaLog_l3"], ["log_N_MA_lag1", "MetaLog_l3"])):
        r0, h0, d0 = fit(p, base + CT)
        key0 = lg.soma(h0, pol)[0] if len(pol) > 1 else h0.params[pol[0]]
        pk0 = lg.soma(h0, pol)[2] if len(pol) > 1 else h0.pvalues[pol[0]]
        add(f"{fam} base", coef=float(key0), p=float(pk0), R2aj=r0.rsquared_adj, AIC=r0.aic, BIC=r0.bic, n=int(r0.nobs))
        # 1 sazonalidade
        r1, h1, d1 = fit(p, base + CT + meses)
        f = r1.compare_f_test(sm.OLS(d1["log_N_MA"], sm.add_constant(d1[base + CT])).fit())
        fz = h1.wald_test(" , ".join(f"{m} = 0" for m in meses), use_f=True, scalar=True)
        key = lg.soma(h1, pol)[0] if len(pol) > 1 else h1.params[pol[0]]
        pk = lg.soma(h1, pol)[2] if len(pol) > 1 else h1.pvalues[pol[0]]
        add(f"{fam} + dummies de mês", coef=float(key), p=float(pk), F_meses_p_HAC=float(fz.pvalue), F_meses_p_classico=float(f[1]), R2aj=r1.rsquared_adj, AIC=r1.aic, BIC=r1.bic)
        # 2 defasagens da dependente
        for L in (2, 3):
            xs = base + [f"log_N_MA_lag{k}" for k in range(2, L + 1)] + CT
            r2, h2, d2 = fit(p, xs)
            key = lg.soma(h2, pol)[0] if len(pol) > 1 else h2.params[pol[0]]
            pk = lg.soma(h2, pol)[2] if len(pol) > 1 else h2.pvalues[pol[0]]
            add(f"{fam} AR({L}) na dependente", coef=float(key), p=float(pk), AIC=r2.aic, BIC=r2.bic, BG6=float(acorr_breusch_godfrey(r2, nlags=6)[1]))
        # 3 controles extras
        ex = [c for c in ["IPCA_log", "dIIE_log"] if c in p]
        r3, h3, d3 = fit(p, base + CT + ex)
        key = lg.soma(h3, pol)[0] if len(pol) > 1 else h3.params[pol[0]]
        pk = lg.soma(h3, pol)[2] if len(pol) > 1 else h3.pvalues[pol[0]]
        add(f"{fam} + IPCA_log + ΔIIE_log", coef=float(key), p=float(pk), AIC=r3.aic, BIC=r3.bic, p_IPCA=float(h3.pvalues["IPCA_log"]), p_IIE=float(h3.pvalues["dIIE_log"]))
        # 4 tendência
        r4, h4, d4 = fit(p, base + CT + ["tendencia"])
        key = lg.soma(h4, pol)[0] if len(pol) > 1 else h4.params[pol[0]]
        pk = lg.soma(h4, pol)[2] if len(pol) > 1 else h4.pvalues[pol[0]]
        add(f"{fam} + tendência linear", coef=float(key), p=float(pk), p_tend=float(h4.pvalues["tendencia"]), AIC=r4.aic, BIC=r4.bic)
        # 5 quebra estrutural (Chow nos coeficientes de política) 2015-01 e 2020-03
        for corte in ("2015-01", "2020-03"):
            d = d0.copy(); dummy = (d.index >= pd.Period(corte)).astype(int)
            for c in pol: d[c + "_x"] = d[c] * dummy
            xs = base + CT + [c + "_x" for c in pol]
            X = sm.add_constant(d[xs]); h = sm.OLS(d["log_N_MA"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True}, use_t=True)
            w = h.wald_test(", ".join(f"{c}_x = 0" for c in pol), use_f=True, scalar=True)
            add(f"{fam} quebra em {corte} (interação da política)", p_quebra=float(w.pvalue))
    # 6 nível e variação juntos e janela de defasagens do nível por BIC
    for k in range(0, 13):
        p[f"MetaLog_l{k}"] = p["Meta_log"].shift(k)
    best = []
    for k in range(0, 13):
        r, h, d = fit(p, ["log_N_MA_lag1", f"MetaLog_l{k}"] + CT)
        best.append((k, r.bic, h.params[f"MetaLog_l{k}"], h.pvalues[f"MetaLog_l{k}"]))
    b = pd.DataFrame(best, columns=["k", "BIC", "coef", "p"])
    b.to_csv(SAIDA / "nivel_por_defasagem_BIC.csv", index=False)
    print(b.round(4).to_string())
    # 7 previsão fora da amostra (janela expansiva, 1 passo, começa em 2015-01). Str/Covid ficam de fora: são
    # constantes (=0) antes de 2020 e tornam a regressão não identificada na janela inicial.
    CT2 = ["Ibov_Ret", "dIBC_log"]
    sem = {f"M{m}": f"M{m}" for m in range(2, 13)}
    for nome, xs in (("só AR(1)", ["log_N_MA_lag1"]), ("AR(1) + controles", ["log_N_MA_lag1"] + CT2),
                     ("+ nível Meta k=3", ["log_N_MA_lag1", "MetaLog_l3"] + CT2), ("+ nível Meta k=9", ["log_N_MA_lag1", "MetaLog_l9"] + CT2),
                     ("+ ΔMeta 0-3", ["log_N_MA_lag1"] + POL1 + CT2), ("+ dummies de mês (sem política)", ["log_N_MA_lag1"] + CT2 + meses),
                     ("+ dummies de mês + nível k=9", ["log_N_MA_lag1", "MetaLog_l9"] + CT2 + meses)):
        d = mqo.recorte(p, lg.INI_PADRAO, mqo.FIM).dropna(subset=["log_N_MA"] + xs)
        erros = []
        for i in range(108, len(d)):
            tr, te = d.iloc[:i], d.iloc[i:i + 1]
            m = sm.OLS(tr["log_N_MA"], sm.add_constant(tr[xs])).fit()
            pred = m.predict(sm.add_constant(te[xs], has_constant="add"))
            erros.append(float(te["log_N_MA"].iloc[0] - pred.iloc[0]))
        add(f"Previsão fora da amostra: {nome}", RMSE=float(np.sqrt(np.mean(np.square(erros)))), n_prev=len(erros))
    pd.DataFrame(out).to_csv(SAIDA / "melhorias.csv", index=False)

if __name__ == "__main__":
    main()
