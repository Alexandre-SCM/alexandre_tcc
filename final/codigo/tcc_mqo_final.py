"""
TCC — Taxa Selic e fusões e aquisições no Brasil: evidências sobre o número de operações entre 2006 e 2024
Alexandre Menezes — Ciências Econômicas, Ibmec-DF, 2026. Orientador: Prof. MSc. Laerte Jun Takeuti.

SCRIPT FINAL ÚNICO. Uma execução gera TODAS as tabelas, figuras e estimativas do trabalho em ../resultados.

Modelo principal (equação 1 do TCC), estimado por MQO com erros-padrão HAC de Newey-West (6 defasagens):
  ln(MA_t) = a + rho·ln(MA_{t-1}) + g_k·Selic_{t-k} + b1·RetIbov_t + b2·ΔIBC_t + b3·Covid_t
             + Σ_{m=2..12} d_m·D_{m,t} + e_t,          k ∈ {1, 3, 6, 9, 12}; principal k=6, alternativo k=9
  Selic = 100·ln(1 + Meta/100); RetIbov = 100·Δln(Ibovespa); ΔIBC = 100·Δln(IBC-Br com ajuste sazonal);
  Covid = 1 de mar/2020 a dez/2020; amostra de estimação jan/2006 a dez/2024 (228 meses; 2005 só forma as defasagens).

Testes: ADF e KPSS (com constante); ajuste de Holm nas 5 defasagens; diagnóstico de k=6 e k=9
(Breusch-Godfrey(6), Breusch-Pagan, RESET); F das variáveis de mês.
Robustez (Tabela 6.7): R1 IPCA, ΔIIE-Br e Δcâmbio; R2 sem 2020-2022; R3 subamostra 2015-2024.
Teste de sensibilidade (Tabela 6.8): variação mensal da Selic no lugar do nível (t-1, t-3, t-6; somas 0-3 e 0-6).
Não é confirmação de robustez: os coeficientes não repetem o sinal negativo e quase todos têm autocorrelação.

Dados:
  ../dados/factset/Deals_Results_TCC_FactSet_Att.xlsx  base da FactSet (licença institucional; NÃO versionada)
  ../dados/sgs/sgs_{432,433,24364,3695}.csv              séries do BCB/SGS usadas na execução final
  ../dados/macro_controls.csv                            Ibovespa (B3) e IIE-Br (FGV IBRE), mensais

Rodar (na pasta final/codigo):  python tcc_mqo_final.py            (usa as séries do SGS versionadas)
                                python tcc_mqo_final.py --atualizar (baixa o SGS de novo; os números podem
                                                                     mudar se o BCB tiver revisado as séries)
"""
import json
import sys
import time
import urllib.request
import warnings
from datetime import date
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
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE = Path(__file__).resolve().parent            # final/codigo
DADOS = BASE.parent / "dados"
ARQUIVO = DADOS / "factset" / "Deals_Results_TCC_FactSet_Att.xlsx"
MACRO_CSV = DADOS / "macro_controls.csv"
CACHE = DADOS / "sgs"
SAIDA = BASE.parent / "resultados"

EXIGIR_AO_VIVO = False
ATUALIZAR_SGS = "--atualizar" in sys.argv   # padrão: usa as séries versionadas (reprodução exata)
SGS_INICIO_ANO = 2003
SGS = {"meta_selic": 432, "ipca": 433, "ibc_br_sa": 24364, "cambio": 3695}

DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0
IDS_DUPLICADOS = ["706879MM", "734331MM"]

INICIO, FIM = "2005-01", "2024-12"
INI_PADRAO = "2006-01"
COVID_INICIO, COVID_FIM = "2020-03", "2020-12"
NW_LAGS = 6
MAX_LAG = 12

LAGS = (1, 3, 6, 9, 12)
K_PRINCIPAL, K_ALTERNATIVO = 6, 9
CTRL = ["Ibov_Ret", "dIBC_log", "Covid"]
MESES = [f"M{m}" for m in range(2, 13)]
AR1 = ["log_N_MA_lag1"]

FONTES: dict = {}
L100 = lambda taxa: 100 * np.log1p(taxa / 100.0)          # taxa em % -> taxa contínua (×100)


# =============================================================================
# 1. SGS/BCB
# =============================================================================
def _get_json(url, tentativas=8, timeout=60):
    """A API do SGS às vezes devolve HTML (erro/bloqueio temporário): nesse caso tenta de novo."""
    ultimo = None
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TCC Ibmec-DF)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"falha ao baixar {url}: {ultimo}")


def baixar_sgs(codigo):
    """Baixa em blocos de 5 anos (a API limita o intervalo das séries diárias)."""
    partes, ano, hoje = [], SGS_INICIO_ANO, date.today()
    while ano <= hoje.year:
        fim_ano = min(ano + 4, hoje.year)
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json"
               f"&dataInicial=01/01/{ano}&dataFinal=31/12/{fim_ano}")
        dados = _get_json(url)
        if dados:
            partes.append(pd.DataFrame(dados))
        ano = fim_ano + 1
    if not partes:
        raise RuntimeError(f"SGS {codigo}: resposta vazia")
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna().drop_duplicates("data").sort_values("data")
    return pd.Series(df["valor"].values, index=pd.DatetimeIndex(df["data"]), name=str(codigo))


def carregar_sgs(nome, codigo):
    """Lê a série versionada em dados/sgs; com --atualizar, baixa de novo da API do BCB e regrava o arquivo."""
    arq = CACHE / f"sgs_{codigo}.csv"
    if not ATUALIZAR_SGS and arq.exists():
        c = pd.read_csv(arq, parse_dates=["data"])
        s = pd.Series(c["valor"].values, index=pd.DatetimeIndex(c["data"])).dropna().sort_index()
        FONTES[nome] = (f"SGS {codigo}", "série versionada em dados/sgs (execução final)", s.index.min(), s.index.max(), len(s))
        return s
    try:
        s = baixar_sgs(codigo)
    except Exception as e:  # noqa: BLE001
        if EXIGIR_AO_VIVO or not arq.exists():
            raise SystemExit(f"[ERRO] {nome} (SGS {codigo}) não foi baixado: {e}")
        c = pd.read_csv(arq, parse_dates=["data"])
        s = pd.Series(c["valor"].values, index=pd.DatetimeIndex(c["data"])).dropna().sort_index()
        FONTES[nome] = (f"SGS {codigo}", f"cache local ./dados_sgs (download falhou: {str(e)[:100]})",
                        s.index.min(), s.index.max(), len(s))
        print(f"  [AVISO] {nome} (SGS {codigo}) veio do cache, não do download ao vivo.")
        return s
    s.rename("valor").rename_axis("data").to_csv(arq)
    FONTES[nome] = (f"SGS {codigo}", "download ao vivo (API BCB)", s.index.min(), s.index.max(), len(s))
    return s


def _mensal(s, como="last"):
    """Série diária ou mensal -> mensal indexada por Period('M')."""
    g = s.groupby(s.index.to_period("M"))
    out = g.last() if como == "last" else g.mean()
    return out.reindex(pd.period_range(out.index.min(), FIM, freq="M")).ffill()


# =============================================================================
# 2. BASE DE NEGÓCIOS E CONTROLES DE MERCADO
# =============================================================================
def carregar_negocios():
    if not ARQUIVO.exists():
        raise SystemExit(f"[ERRO] Base da FactSet não encontrada em {ARQUIVO}.\n"
                         "Ela não é publicada no repositório por restrição de licença; copie o arquivo para essa pasta.")
    df = pd.read_excel(ARQUIVO, sheet_name="Transactions_Results", header=2)
    df["Announcement Date"] = pd.to_datetime(df["Announcement Date"])
    df["valor"] = pd.to_numeric(df["Deal Size (USD, MM)"], errors="coerce")
    df["pct"] = pd.to_numeric(df["% of Target Shares Sought"], errors="coerce")
    return df


def funil(df):
    """Etapas de seleção da amostra (Tabela 5.1)."""
    e0 = df[(df["Deal Status"] == "Completed") & (df["Country (Target/Issuer)"] == "Brazil")]
    e1 = e0[e0["valor"].notna()]
    e2 = e1[e1["valor"] > DEAL_MIN_USD_MM]
    e3 = e2[e2["pct"] > PCT_MIN]
    rep = e3["Deal ID"].isin(IDS_DUPLICADOS) & e3.duplicated(keep="first")
    e4 = e3[~rep]
    etapas = [("Extração: M&A concluídos, alvo no Brasil, 2005 a 2024", e0), ("Valor da operação divulgado", e1),
              ("Valor superior a US$ 0,5 milhão", e2), ("Participação buscada superior a 50% do capital", e3),
              ("Exclusão de registros duplicados", e4)]
    linhas, ant = [], None
    for i, (nome, d) in enumerate(etapas):
        linhas.append({"etapa": i, "criterio": nome, "excluidas": None if ant is None else ant - len(d), "restantes": len(d)})
        ant = len(d)
    return e4.copy(), pd.DataFrame(linhas)


def contagem_mensal(negocios):
    s = negocios["Announcement Date"].dt.to_period("M").value_counts().sort_index()
    return s.reindex(pd.period_range(INICIO, FIM, freq="M"), fill_value=0).rename("N_MA")


def _serie_planilha(m, col_data, col_val, nome):
    s = m.iloc[:, [col_data, col_val]].copy()
    s.columns = ["data", nome]
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    s = s.dropna()
    s.index = s["data"].dt.to_period("M")
    return pd.to_numeric(s[nome], errors="coerce")


def carregar_macro_controls():
    """Ibovespa (B3) e IIE-Br (FGV IBRE). Se o CSV não existir, é extraído da aba Macro_Controls da planilha."""
    if not MACRO_CSV.exists():
        m = pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)
        ibov = _serie_planilha(m, 28, 29, "Ibov_Nivel")
        iie = _serie_planilha(m, 19, 20, "IIE_Br")
        out = pd.concat([ibov[~ibov.index.duplicated()], iie[~iie.index.duplicated()]], axis=1)
        out.index.name = "mes"
        out.to_csv(MACRO_CSV)
    c = pd.read_csv(MACRO_CSV)
    c.index = pd.PeriodIndex(c["mes"], freq="M")
    for nome, org in (("Ibov_Nivel", "B3"), ("IIE_Br", "FGV IBRE")):
        s = c[nome].dropna()
        FONTES[nome] = (org, "dados/macro_controls.csv", s.index.min().to_timestamp(), s.index.max().to_timestamp(), len(s))
    return c["Ibov_Nivel"], c["IIE_Br"]


def montar_painel(negocios):
    meta = carregar_sgs("meta_selic", SGS["meta_selic"])
    ipca = carregar_sgs("ipca", SGS["ipca"])
    ibc = carregar_sgs("ibc_br_sa", SGS["ibc_br_sa"])
    cambio = carregar_sgs("cambio", SGS["cambio"])
    macro = pd.concat({"Meta": _mensal(meta), "IPCA": _mensal(ipca), "IBC_Br": _mensal(ibc),
                       "Cambio": _mensal(cambio)}, axis=1)
    ibov, iie = carregar_macro_controls()
    macro = macro.join(ibov.rename("Ibov_Nivel"), how="outer").join(iie.rename("IIE_Br"), how="outer")

    idx = pd.period_range("2003-01", FIM, freq="M")
    p = macro.reindex(idx)
    p["N_MA"] = contagem_mensal(negocios).reindex(idx)
    p["log_N_MA"] = np.log(p["N_MA"])
    p["log_N_MA_lag1"] = p["log_N_MA"].shift(1)
    p["Meta_log"] = L100(p["Meta"])
    p["dMeta_log"] = p["Meta_log"].diff()
    for k in range(MAX_LAG + 1):
        p[f"MetaLog_l{k}"] = p["Meta_log"].shift(k)
        p[f"dMetaLog_l{k}"] = p["dMeta_log"].shift(k)
    p["Ibov_Ret"] = 100 * np.log(p["Ibov_Nivel"]).diff()
    p["dIBC_log"] = 100 * np.log(p["IBC_Br"]).diff()
    p["IPCA_log"] = L100(p["IPCA"])
    p["dCambio_log"] = 100 * np.log(p["Cambio"]).diff()
    p["dIIE_log"] = 100 * np.log(p["IIE_Br"]).diff()
    s = p.index.astype(str)
    p["Covid"] = ((s >= COVID_INICIO) & (s <= COVID_FIM)).astype(int)
    for mes in range(2, 13):
        p[f"M{mes}"] = (p.index.month == mes).astype(int)
    p.index.name = "Periodo"
    return p


# =============================================================================
# 3. ESTIMAÇÃO E DIAGNÓSTICO
# =============================================================================
def recorte(p, ini, fim):
    return p.loc[ini:fim] if ini else p.loc[:fim]


def ajuste(p, xs, ini=INI_PADRAO, fim=FIM, excluir_anos=None):
    d = recorte(p, ini, fim).dropna(subset=["log_N_MA"] + xs)
    if excluir_anos:
        d = d[~d.index.year.isin(excluir_anos)]
    xs = [x for x in xs if d[x].nunique() > 1]
    X = sm.add_constant(d[xs])
    plain = sm.OLS(d["log_N_MA"], X).fit()
    hac = sm.OLS(d["log_N_MA"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS, "use_correction": True},
                                       use_t=True)
    return hac, plain, d, xs


def diag(plain, hac, meses_in):
    r = {"R2_aj": plain.rsquared_adj, "AIC": plain.aic, "BIC": plain.bic,
         "BG(6) p": acorr_breusch_godfrey(plain, nlags=6)[1], "JB p": jarque_bera(plain.resid)[1],
         "assimetria": jarque_bera(plain.resid)[2], "curtose": jarque_bera(plain.resid)[3],
         "BP p": het_breuschpagan(plain.resid, plain.model.exog)[1],
         "RESET p": float(linear_reset(plain, power=2, use_f=True).pvalue)}
    if meses_in:
        r["F meses p (HAC)"] = float(hac.wald_test(", ".join(f"{m} = 0" for m in meses_in), use_f=True,
                                                   scalar=True).pvalue)
    return r


def chave(hac, pol):
    if len(pol) > 1:
        t = hac.t_test(" + ".join(pol) + " = 0")
        b, ep, pv = float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))
        lo, hi = [float(x) for x in np.squeeze(t.conf_int())]
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
    return r, hac, plain


# =============================================================================
# 4. TABELAS E FIGURAS DO TCC
# =============================================================================
SETORES_PT = {"Finance": "Finanças", "Technology Services": "Serviços de tecnologia",
              "Process Industries": "Indústrias de processo", "Utilities": "Utilidade pública",
              "Non-Energy Minerals": "Minerais não energéticos", "Commercial Services": "Serviços comerciais",
              "Consumer Services": "Serviços ao consumidor", "Health Services": "Serviços de saúde"}
TIPOS_PT = {"Acquisition / Merger": "Aquisição ou fusão", "Majority Stake": "Participação majoritária",
            "Spinoff": "Cisão"}
CASOS = [("2015-08", "HSBC"), ("2016-07", "CPFL"), ("2021-01", "Notre Dame"), ("2021-07", "V.tal"), ("2024-01", "Soma")]
SERIES_ESTAC = [("ln(MA)", "log_N_MA"), ("Selic (nível)", "Meta_log"), ("Variação da Selic", "dMeta_log"),
                ("RetIbov", "Ibov_Ret"), ("ΔIBC", "dIBC_log")]


def salvar(df, nome, **kw):
    df.to_csv(SAIDA / nome, **kw)
    return df


def capitulo2(neg, p):
    n = neg[neg["Announcement Date"] >= "2006-01-01"]
    N = len(n)
    sel = p.loc["2006-01":"2024-12", "Meta"]
    a = n.groupby(n["Announcement Date"].dt.year).agg(operacoes=("valor", "size"), valor_total_usd_bi=("valor", "sum"),
                                                     valor_mediano_usd_mi=("valor", "median"))
    a["valor_total_usd_bi"] /= 1000
    a["selic_fim_do_ano"] = sel.groupby(sel.index.year).last()
    a["selic_media_do_ano"] = sel.groupby(sel.index.year).mean()
    a.loc["Total"] = [N, n["valor"].sum() / 1000, n["valor"].median(), np.nan, np.nan]
    salvar(a, "tab_2_1_operacoes_por_ano.csv")
    faixas = pd.cut(n["valor"], [0, 10, 50, 100, 500, 1e9],
                    labels=["Até 10", "Acima de 10 até 50", "Acima de 50 até 100", "Acima de 100 até 500", "Acima de 500"])
    t = n["Transaction Type"].map(TIPOS_PT)
    porte = pd.concat([faixas.value_counts().sort_index().rename("operacoes").to_frame().assign(grupo="Faixa de valor (US$ milhões)"),
                       t.value_counts().rename("operacoes").to_frame().assign(grupo="Tipo de operação")])
    porte["pct_total"] = 100 * porte["operacoes"] / N
    salvar(porte, "tab_2_2_porte_e_tipo.csv")
    s = n["FactSet Sector (Target/Issuer)"].value_counts()
    top = s.head(8)
    setor = pd.DataFrame({"operacoes": top.values}, index=[SETORES_PT.get(k, k) for k in top.index])
    setor.loc[f"Demais setores ({len(s) - 8})"] = N - top.sum()
    setor["pct_total"] = 100 * setor["operacoes"] / N
    salvar(setor, "tab_2_3_setores.csv")
    q = p.copy()
    q["meta_6m_antes"] = q["Meta"].shift(6)
    q = q.loc["2006-01":"2024-12"]
    q["faixa"] = pd.cut(q["meta_6m_antes"], [0, 7, 11, 14, 20],
                        labels=["Até 7%", "Acima de 7% até 11%", "Acima de 11% até 14%", "Acima de 14%"])
    fx = q.groupby("faixa", observed=True)["N_MA"].agg(meses="size", operacoes="sum", media_mensal="mean", mediana_mensal="median")
    salvar(fx, "tab_2_4_operacoes_por_faixa_da_selic.csv")
    casos = []
    for m, chave_ in CASOS:
        d = n[(n["Announcement Date"].dt.strftime("%Y-%m") == m) & n["Company Name (Target/Issuer)"].str.contains(chave_)]
        r = d.iloc[0]
        casos.append({"anuncio": m, "alvo": r["Company Name (Target/Issuer)"], "comprador": r["Company Name (Buyer/Investor)"],
                      "setor": SETORES_PT.get(r["FactSet Sector (Target/Issuer)"], r["FactSet Sector (Target/Issuer)"]),
                      "valor_usd_bi": r["valor"] / 1000, "selic_no_mes": p.loc[m, "Meta"]})
    salvar(pd.DataFrame(casos), "quadro_2_1_operacoes_ilustrativas.csv", index=False)
    mes = n["Announcement Date"].dt.month.value_counts().sort_index()
    salvar(pd.DataFrame({"operacoes": mes, "pct_total": 100 * mes / N}), "sazonalidade_por_mes.csv")
    # Figura 2.1
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    anos = a.drop("Total")
    ax.bar(anos.index.astype(int), anos["operacoes"], color="#1f3864", alpha=0.85, label="Número de operações (eixo esquerdo)")
    ax.set_ylabel("Número de operações"); ax.set_xticks(anos.index.astype(int)); ax.tick_params(axis="x", rotation=45)
    ax2 = ax.twinx()
    ax2.plot(anos.index.astype(int), anos["selic_media_do_ano"], color="#b7472a", marker="o", lw=1.6,
             label="Meta da Selic, média do ano (eixo direito)")
    ax2.set_ylabel("% ao ano"); ax2.set_ylim(0, 18)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Operações de M&A no Brasil por ano e meta média da Selic (2006-2024)"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(SAIDA / "fig_2_1_operacoes_por_ano.png"); plt.close(fig)


def figura_6_1(p):
    d = p.loc[INICIO:FIM]
    x = d.index.to_timestamp()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)
    ax[0].plot(x, d["N_MA"], color="#1f3864", lw=1)
    ax[0].set_title("Nº mensal de operações de M&A (amostra filtrada)")
    ax[1].step(x, d["Meta"], color="#b7472a", lw=1.4, where="post")
    ax[1].set_title("Meta da Selic (SGS 432), fim do mês, % a.a.")
    for a_ in ax:
        a_.axvspan(pd.Timestamp("2005-01-01"), pd.Timestamp("2006-01-01"), color="#bbbbbb", alpha=0.35, lw=0)
        a_.grid(alpha=0.25)
    ax[0].text(pd.Timestamp("2005-02-01"), ax[0].get_ylim()[1] * 0.9, "2005: só\ndefasagens", fontsize=7, va="top")
    fig.tight_layout(); fig.savefig(SAIDA / "fig_6_1_operacoes_e_selic_mensal.png"); plt.close(fig)


def main():
    pd.set_option("display.width", 230, "display.max_columns", 30)
    SAIDA.mkdir(exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    bruto = carregar_negocios()
    negocios, fn = funil(bruto)
    salvar(fn, "tab_5_1_selecao_da_amostra.csv", index=False)
    print(fn.to_string(index=False))
    p = montar_painel(negocios)
    pd.DataFrame(FONTES, index=["série", "origem usada", "início", "fim", "n obs."]).T.to_csv(SAIDA / "fontes_dados.csv")
    w = p.loc[INI_PADRAO:FIM]
    salvar(w, "painel_mensal.csv")

    capitulo2(negocios, p)
    figura_6_1(p)

    # 6.1 descritivas, correlações, estacionariedade
    cols = ["N_MA", "log_N_MA", "Meta", "Meta_log", "dMeta_log", "Ibov_Ret", "dIBC_log", "Covid",
            "IPCA_log", "dIIE_log", "dCambio_log"]
    salvar(w[cols].describe().T[["mean", "50%", "std", "min", "max"]], "tab_6_1_descritivas.csv")
    salvar(w[["log_N_MA", "log_N_MA_lag1", "MetaLog_l6", "MetaLog_l9", "Ibov_Ret", "dIBC_log", "Covid"]].corr(),
           "tab_6_2_correlacoes.csv")
    U = []
    for rot, c in SERIES_ESTAC:
        s = w[c].dropna()
        pa = adfuller(s, regression="c", autolag="AIC")[1]
        pk = kpss(s, regression="c", nlags="auto")[1]
        U.append({"variavel": rot, "ADF p": pa, "KPSS p (limitado a 0,10)": pk,
                  "resultado_5%": "Estacionária" if (pa < 0.05 and pk > 0.05) else "Inconclusivo"})
    print("\n== Estacionariedade ==\n", salvar(pd.DataFrame(U), "tab_6_3_estacionariedade.csv", index=False).round(3).to_string(index=False))

    # 6.2 resultado principal
    prin, hacs, plains = [], {}, {}
    for k in LAGS:
        r, h, pl = linha(f"Selic nível, k={k}", p, [f"MetaLog_l{k}"], CTRL)
        r["k"] = k; prin.append(r); hacs[k], plains[k] = h, pl
    P = pd.DataFrame(prin)
    P["p Holm"] = multipletests(P["p"], method="holm")[1]
    P["variacao_pct_aprox"] = 100 * (np.exp(P["coef"]) - 1)
    salvar(P, "tab_6_4_selic_por_defasagem.csv", index=False)
    print("\n== Principal ==\n", P[["k", "coef", "ep_HAC", "p", "p Holm", "ic95_inf", "ic95_sup", "R2_aj", "variacao_pct_aprox"]].round(4).to_string(index=False))
    t5 = hacs[K_PRINCIPAL].summary2().tables[1]
    salvar(t5, f"tab_6_5_coeficientes_k{K_PRINCIPAL}.csv")
    D = P[P["k"].isin([K_PRINCIPAL, K_ALTERNATIVO])][["k", "BG(6) p", "BP p", "RESET p", "JB p", "F meses p (HAC)"]]
    salvar(D, "tab_6_6_diagnostico_k6_k9.csv", index=False)
    print("\n== Diagnóstico ==\n", D.round(3).to_string(index=False))

    # 6.3 robustez e sensibilidade
    R = [linha("Principal", p, ["MetaLog_l6"], CTRL)[0],
         linha("R1 + IPCA, ΔIIE-Br, Δcâmbio", p, ["MetaLog_l6"], CTRL + ["IPCA_log", "dIIE_log", "dCambio_log"])[0],
         linha("R2 sem 2020-2022", p, ["MetaLog_l6"], CTRL, excluir_anos=[2020, 2021, 2022])[0],
         linha("R3 2015-2024", p, ["MetaLog_l6"], CTRL, ini="2015-01")[0]]
    salvar(pd.DataFrame(R), "tab_6_7_robustez.csv", index=False)
    S = [linha(f"Variação em t-{k}", p, [f"dMetaLog_l{k}"], CTRL)[0] for k in (1, 3, 6)]
    S += [linha("Soma das variações de t a t-3", p, [f"dMetaLog_l{k}" for k in range(4)], CTRL)[0],
          linha("Soma das variações de t a t-6", p, [f"dMetaLog_l{k}" for k in range(7)], CTRL)[0]]
    salvar(pd.DataFrame(S), "tab_6_8_sensibilidade_variacao_selic.csv", index=False)
    print("\n== Robustez e sensibilidade ==\n", pd.DataFrame(R + S)[["modelo", "coef", "ep_HAC", "p", "n", "R2_aj", "BG(6) p", "BP p"]].round(4).to_string())

    # Figuras 6.2 e 6.3
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    ax.errorbar(P["k"], P["coef"], yerr=[P["coef"] - P["ic95_inf"], P["ic95_sup"] - P["coef"]], fmt="s-",
                color="#1f3864", capsize=4)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xticks(list(LAGS))
    ax.set_xlabel("Defasagem k (meses)"); ax.set_ylabel("Coeficiente da Selic sobre ln(M&A)")
    ax.set_title("Coeficiente da Selic sobre M&A, por defasagem (IC 95% HAC)"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(SAIDA / "fig_6_2_coeficiente_por_defasagem.png"); plt.close(fig)
    h = plains[K_PRINCIPAL]; idx = h.fittedvalues.index.to_timestamp(); res = h.resid
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), dpi=150)
    ax[0].plot(idx, h.model.endog, color="#555555", lw=1.2, label="ln(M&A) observado")
    ax[0].plot(idx, h.fittedvalues, color="#1f3864", lw=1.2, label=f"ajustado (k={K_PRINCIPAL})")
    ax[0].set_title(f"Modelo k={K_PRINCIPAL}: observado e ajustado"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    ax[1].plot(idx, res, color="#1f3864", lw=1); ax[1].axhline(0, color="k", lw=0.8)
    for sgn in (2, -2):
        ax[1].axhline(sgn * res.std(), color="#d62728", lw=0.8, ls="--")
    ax[1].set_title("Resíduos"); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(SAIDA / "fig_6_3_ajuste_e_residuos_k6.png"); plt.close(fig)

    with open(SAIDA / "execucao.json", "w", encoding="utf-8") as f:
        json.dump({"data_execucao": date.today().isoformat(), "atualizou_sgs": ATUALIZAR_SGS,
                   "operacoes_amostra": int(len(negocios)), "meses_estimacao": int(hacs[K_PRINCIPAL].nobs),
                   "coef_k6": float(P.loc[P.k == 6, "coef"].iloc[0])}, f, ensure_ascii=False, indent=1)
    print("\nPronto. Saída em", SAIDA)


if __name__ == "__main__":
    main()
