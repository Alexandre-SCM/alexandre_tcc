"""
MODELO FINAL DO TCC — MQO mensal em log, índice de rigor (Stringency), dummies de mês
====================================================================================
Script único (não importa nenhum outro .py do projeto): carrega a base, aplica os filtros,
baixa as séries do SGS/BCB, monta o painel, estima, testa e salva tudo em ./resultados_mqo_final.

Especificação principal (H1 — política monetária em NÍVEL):

  ln(M&A)_t = c + a·ln(M&A)_{t-1} + g_k·Meta_log_{t-k} + b1·Ibov_Ret_t + b2·ΔIBC_log_t
              + b3·Stringency_t + Σ_{m=2..12} d_m·Mês_m + e_t          k ∈ {1, 3, 6, 9, 12}

  Meta_log = 100·ln(1 + Meta Selic/100);  Ibov_Ret = 100·Δln(Ibovespa);  ΔIBC_log = 100·Δln(IBC-Br SA);
  Stringency = índice de rigor da Oxford (OxCGRT, Brasil, média mensal/100; 0 antes de 2020;
  2023+ repete o último valor, 22,2); erros-padrão HAC (Newey-West, 6 defasagens);
  amostra jan/2006 a dez/2024. Principal k=6; alternativo k=9. Sem a expectativa da Selic.

Dados:
  - Deals_Results_TCC_FactSet_Att.xlsx (aba Transactions_Results): filtros Completed + alvo Brazil +
    Deal Size > US$ 0,5 MM + % adquirida > 50%; uma ocorrência dos IDs duplicados 706879MM e 734331MM
    (2.292 -> 2.290 operações).
  - SGS/BCB, download ao vivo: 432 (Meta Selic), 433 (IPCA), 24364 (IBC-Br dessaz.), 3695 (câmbio).
    A origem de cada série fica em fontes_dados.csv. Com EXIGIR_AO_VIVO = True, o script para se
    algum download falhar (nunca usa cache ou planilha em silêncio).
  - Ibovespa (B3) e IIE-Br (FGV): aba Macro_Controls da planilha (não são do SGS).
  - Stringency_index.csv (OxCGRT) em ./ ou ./dados_covid/; se ausente, usa o mensal já extraído
    dele em ./dados_covid/stringency_brasil_mensal.csv.

Robustez:
  R1  variação da Meta (ΔMeta_log), efeitos pontuais k=1,3,6 e acumulados 0-3, 0-6
  R2  controles extras (IPCA_log, ΔIIE_log, ΔCâmbio_log)
  R3  dummy Covid no lugar do Stringency
  R4  sem dummies de mês
  R5  subamostras (2015-2024; sem 2020-2022; até 2019)
  R6  estimadores alternativos no k=6 (HC3, Poisson+HAC, Huber, mediana)
  R7  avaliação fora da amostra (janela expansiva), com e sem a política. Ibov_Ret e ΔIBC_log são
      do próprio mês: é uma avaliação com dados já conhecidos, não uma previsão feita antecipadamente.

Diagnóstico de cada modelo: Breusch-Godfrey(6), Jarque-Bera, Breusch-Pagan, RESET, F das dummies de mês.

Rodar: python Código_TCC_MQO_FINAL.py
"""

import json
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

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados_mqo_final"
CACHE = BASE / "dados_sgs"
DIR_COVID = BASE / "dados_covid"
STR_BRUTO = [BASE / "Stringency_index.csv", DIR_COVID / "Stringency_index.csv"]
STR_MENSAL = DIR_COVID / "stringency_brasil_mensal.csv"
SAIDA.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

EXIGIR_AO_VIVO = True          # True = para se o download do SGS falhar
SGS_INICIO_ANO = 2003          # desde 2003 para ter as defasagens já em 2005
SGS = {"meta_selic": 432, "ipca": 433, "ibc_br_sa": 24364, "cambio": 3695}

DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0
IDS_DUPLICADOS = ["706879MM", "734331MM"]   # linhas idênticas conferidas na fonte

INICIO, FIM = "2005-01", "2024-12"
INI_PADRAO = "2006-01"         # defasagens de até 12 meses exigem começar em 2006-01
COVID_INICIO, COVID_FIM = "2020-03", "2020-12"
NW_LAGS = 6
MAX_LAG = 12

LAGS = (1, 3, 6, 9, 12)
K_PRINCIPAL, K_ALTERNATIVO = 6, 9
CTRL = ["Ibov_Ret", "dIBC_log", "Str"]
MESES = [f"M{m}" for m in range(2, 13)]
AR1 = ["log_N_MA_lag1"]

FONTES: dict = {}
L100 = lambda taxa: 100 * np.log1p(taxa / 100.0)          # taxa em % -> log-taxa (×100)


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
    """Download ao vivo; o resultado é gravado em ./dados_sgs. Só usa o cache se EXIGIR_AO_VIVO = False."""
    arq = CACHE / f"sgs_{codigo}.csv"
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
# 2. BASE DE NEGÓCIOS, PLANILHA E STRINGENCY
# =============================================================================
def carregar_negocios():
    df = pd.read_excel(ARQUIVO, sheet_name="Transactions_Results", header=2)
    df["Announcement Date"] = pd.to_datetime(df["Announcement Date"])
    df["valor"] = pd.to_numeric(df["Deal Size (USD, MM)"], errors="coerce")
    df["pct"] = pd.to_numeric(df["% of Target Shares Sought"], errors="coerce")
    return df


def aplicar_filtros(df):
    m = ((df["Deal Status"] == "Completed") & (df["Country (Target/Issuer)"] == "Brazil")
         & (df["valor"] > DEAL_MIN_USD_MM) & (df["pct"] > PCT_MIN))
    out = df[m].copy()
    n_antes = len(out)
    # remove só a 2ª ocorrência dos IDs conferidos, e apenas se a linha for idêntica à 1ª
    rep = out["Deal ID"].isin(IDS_DUPLICADOS) & out.duplicated(keep="first")
    out = out[~rep]
    return out, n_antes


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


def stringency_mensal():
    arq = next((a for a in STR_BRUTO if a.exists()), None)
    if arq is not None:
        d = pd.read_csv(arq, sep=";", skiprows=[1], dtype=str,
                        usecols=["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average"])
        b = d[(d.CountryCode == "BRA") & (d.Jurisdiction == "NAT_TOTAL")].copy()
        b["Date"] = pd.to_datetime(b["Date"], format="%Y%m%d")
        b["x"] = pd.to_numeric(b["StringencyIndex_Average"])
        s = b.set_index("Date")["x"].resample("ME").mean()
        s.index = s.index.to_period("M")
        DIR_COVID.mkdir(exist_ok=True)
        s.rename("stringency").to_frame().to_csv(STR_MENSAL)
        origem = f"{arq.name} (OxCGRT, diário -> média mensal)"
    else:
        s = pd.read_csv(STR_MENSAL, index_col=0)["stringency"]
        s.index = pd.PeriodIndex(s.index, freq="M")
        origem = f"{STR_MENSAL.name} (média mensal já extraída de Stringency_index.csv)"
    FONTES["stringency"] = ("OxCGRT", origem, s.index.min().to_timestamp(), s.index.max().to_timestamp(), len(s))
    return s


def montar_painel(negocios):
    m = pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)
    meta = carregar_sgs("meta_selic", SGS["meta_selic"])
    ipca = carregar_sgs("ipca", SGS["ipca"])
    ibc = carregar_sgs("ibc_br_sa", SGS["ibc_br_sa"])
    cambio = carregar_sgs("cambio", SGS["cambio"])
    macro = pd.concat({"Meta": _mensal(meta), "IPCA": _mensal(ipca), "IBC_Br": _mensal(ibc),
                       "Cambio": _mensal(cambio)}, axis=1)

    ibov = _serie_planilha(m, 28, 29, "Ibov_Nivel")
    iie = _serie_planilha(m, 19, 20, "IIE_Br")
    for nome, s, org in (("ibovespa", ibov, "B3"), ("iie_br", iie, "FGV")):
        FONTES[nome] = (f"{org} (planilha)", "planilha Macro_Controls — não é SGS",
                        s.index.min().to_timestamp(), s.index.max().to_timestamp(), len(s))
    macro = macro.join(ibov.rename("Ibov_Nivel"), how="outer").join(iie.rename("IIE_Br"), how="outer")

    idx = pd.period_range("2003-01", FIM, freq="M")
    p = macro.reindex(idx)
    p["N_MA"] = contagem_mensal(negocios).reindex(idx)
    p["log_N_MA"] = np.log(p["N_MA"])
    p["log_N_MA_lag1"] = p["log_N_MA"].shift(1)

    # política monetária, em log
    p["Meta_log"] = L100(p["Meta"])
    p["dMeta_log"] = p["Meta_log"].diff()
    for k in range(MAX_LAG + 1):
        p[f"MetaLog_l{k}"] = p["Meta_log"].shift(k)
        p[f"dMetaLog_l{k}"] = p["dMeta_log"].shift(k)

    # controles
    p["Ibov_Ret"] = 100 * np.log(p["Ibov_Nivel"]).diff()
    p["dIBC_log"] = 100 * np.log(p["IBC_Br"]).diff()
    p["IPCA_log"] = L100(p["IPCA"])
    p["dCambio_log"] = 100 * np.log(p["Cambio"]).diff()
    p["dIIE_log"] = 100 * np.log(p["IIE_Br"]).diff()
    s = p.index.astype(str)
    p["Covid"] = ((s >= COVID_INICIO) & (s <= COVID_FIM)).astype(int)

    st = stringency_mensal() / 100.0
    p["Str"] = st.reindex(p.index)
    p.loc[p.index < st.index.min(), "Str"] = 0.0
    p.loc[p.index > st.index.max(), "Str"] = st.iloc[-1]       # 2023+ repete o último valor (22,2)

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
         "BP p": het_breuschpagan(plain.resid, plain.model.exog)[1]}
    try:
        r["RESET p"] = float(linear_reset(plain, power=2, use_f=True).pvalue)
    except Exception:  # noqa: BLE001
        r["RESET p"] = np.nan
    if meses_in:
        r["F meses p (HAC)"] = float(hac.wald_test(", ".join(f"{m} = 0" for m in meses_in), use_f=True,
                                                   scalar=True).pvalue)
    return r


def diag_detalhado(plain):
    """Estatísticas e p-valores dos quatro testes, para a tabela de diagnóstico."""
    bg = acorr_breusch_godfrey(plain, nlags=6)
    jb = jarque_bera(plain.resid)
    bp = het_breuschpagan(plain.resid, plain.model.exog)
    rs = linear_reset(plain, power=2, use_f=True)
    return {"BG(6) LM": bg[0], "BG(6) p": bg[1], "JB estat.": jb[0], "JB p": jb[1],
            "assimetria": jb[2], "curtose": jb[3], "BP LM": bp[0], "BP p": bp[1],
            "RESET F": float(rs.fvalue), "RESET p": float(rs.pvalue)}


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
# 4. EXECUÇÃO
# =============================================================================
def main():
    pd.set_option("display.width", 230, "display.max_columns", 30)

    print("=" * 74)
    print("1. DADOS")
    bruto = carregar_negocios()
    negocios, n_antes = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} | após filtros: {n_antes:,} | sem duplicidades "
          f"{', '.join(IDS_DUPLICADOS)}: {len(negocios):,}")
    p = montar_painel(negocios)

    fontes = pd.DataFrame(FONTES, index=["série", "origem usada", "início", "fim", "n obs."]).T
    fontes.to_csv(SAIDA / "fontes_dados.csv")
    print("\nDe onde veio cada série:")
    print(fontes.to_string())
    cols_painel = ["N_MA", "log_N_MA", "log_N_MA_lag1", "Meta", "Meta_log", "dMeta_log", "Ibov_Nivel", "Ibov_Ret",
                   "IBC_Br", "dIBC_log", "Str", "Covid", "IPCA", "IPCA_log", "Cambio", "dCambio_log", "IIE_Br",
                   "dIIE_log"]
    p.loc[INICIO:FIM, cols_painel].to_csv(SAIDA / "painel_mensal.csv")
    pd.DataFrame({"etapa": ["base bruta", "após filtros", "sem duplicidades"],
                  "operações": [len(bruto), n_antes, len(negocios)]}).to_csv(SAIDA / "amostra.csv", index=False)

    # ---------- principal ----------
    prin, hacs, plains = [], {}, {}
    for k in LAGS:
        r, h, pl = linha(f"Nível Meta_log, k={k}", p, [f"MetaLog_l{k}"], CTRL)
        r["k"] = k
        r["coef Stringency"] = h.params["Str"]
        r["p Stringency"] = h.pvalues["Str"]
        prin.append(r)
        hacs[k], plains[k] = h, pl
    P = pd.DataFrame(prin)
    P["p_Holm"] = multipletests(P["p"], method="holm")[1]
    P.to_csv(SAIDA / "principal_nivel_por_defasagem.csv", index=False)
    print("\n== PRINCIPAL: nível da Meta (log), com dummies de mês e Stringency ==")
    print(P[["modelo", "coef", "ep_HAC", "p", "p_Holm", "ic95_inf", "ic95_sup", "n", "R2_aj", "AIC", "BIC"]]
          .round(4).to_string())

    D = pd.DataFrame([{"k": k, "n": int(plains[k].nobs), **diag_detalhado(plains[k]),
                       "F meses p (HAC)": P.loc[P.k == k, "F meses p (HAC)"].iloc[0]} for k in LAGS])
    D.to_csv(SAIDA / "diagnosticos_por_defasagem.csv", index=False)
    print("\n== DIAGNÓSTICOS por defasagem (resíduos do MQO) ==")
    print(D.round(4).to_string(index=False))

    for k in (K_PRINCIPAL, K_ALTERNATIVO):
        hacs[k].summary2().tables[1].to_csv(SAIDA / f"coeficientes_k{k}.csv")
        print(f"\nCoeficientes (k={k}):")
        print(hacs[k].summary2().tables[1].round(4).to_string())

    # ---------- robustez ----------
    R = []
    for k in (1, 3, 6):
        R.append(linha(f"R1 ΔMeta_log pontual, k={k}", p, [f"dMetaLog_l{k}"], CTRL)[0])
    R.append(linha("R1 ΔMeta_log acumulado 0-3", p, [f"dMetaLog_l{k}" for k in range(4)], CTRL)[0])
    R.append(linha("R1 ΔMeta_log acumulado 0-6", p, [f"dMetaLog_l{k}" for k in range(7)], CTRL)[0])
    ex = ["IPCA_log", "dIIE_log", "dCambio_log"]
    R.append(linha(f"R2 nível k=6 + {', '.join(ex)}", p, ["MetaLog_l6"], CTRL + ex)[0])
    R.append(linha("R3 nível k=6, dummy Covid no lugar do Stringency", p, ["MetaLog_l6"],
                   ["Ibov_Ret", "dIBC_log", "Covid"])[0])
    R.append(linha("R4 nível k=6, sem dummies de mês", p, ["MetaLog_l6"], CTRL, meses=False)[0])
    R.append(linha("R5 nível k=6, 2015-2024", p, ["MetaLog_l6"], CTRL, ini="2015-01")[0])
    R.append(linha("R5 nível k=6, sem 2020-2022", p, ["MetaLog_l6"], CTRL, excluir_anos=[2020, 2021, 2022])[0])
    R.append(linha("R5 nível k=6, até 2019", p, ["MetaLog_l6"], CTRL, fim="2019-12")[0])
    RB = pd.DataFrame(R)
    RB.to_csv(SAIDA / "robustez.csv", index=False)
    print("\n== ROBUSTEZ ==")
    print(RB[["modelo", "coef", "ep_HAC", "p", "n", "R2_aj", "BG(6) p", "JB p", "BP p", "RESET p"]].round(4).to_string())

    # ---------- estimadores alternativos (k=6) ----------
    xs = AR1 + ["MetaLog_l6"] + CTRL + MESES
    d = recorte(p, INI_PADRAO, FIM).dropna(subset=["log_N_MA", "N_MA"] + xs)
    X = sm.add_constant(d[xs])
    alt = []

    def add(nome, res, var="MetaLog_l6"):
        ic = res.conf_int().loc[var]
        alt.append({"método": nome, "coef": float(res.params[var]), "ep": float(res.bse[var]),
                    "p": float(res.pvalues[var]), "ic95_inf": float(ic.iloc[0]), "ic95_sup": float(ic.iloc[1])})
    add("MQO + HAC (padrão)", sm.OLS(d["log_N_MA"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS,
                                                                                   "use_correction": True}, use_t=True))
    add("MQO + HC3", sm.OLS(d["log_N_MA"], X).fit(cov_type="HC3"))
    add("Poisson (QML) + HAC, usa a contagem", sm.GLM(d["N_MA"], X, family=sm.families.Poisson())
        .fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}))
    add("Regressão robusta (Huber)", sm.RLM(d["log_N_MA"], X, M=sm.robust.norms.HuberT()).fit())
    add("Regressão mediana", sm.QuantReg(d["log_N_MA"], X).fit(q=0.5))
    A = pd.DataFrame(alt)
    A.to_csv(SAIDA / "estimadores_alternativos_k6.csv", index=False)
    print("\n== ESTIMADORES ALTERNATIVOS (k=6) ==")
    print(A.round(4).to_string())

    # ---------- avaliação fora da amostra (dados já conhecidos) ----------
    CT2 = ["Ibov_Ret", "dIBC_log"]
    oos = []
    for nome, xs in (("AR(1)", AR1), ("AR(1) + controles", AR1 + CT2), ("+ dummies de mês", AR1 + CT2 + MESES),
                     ("+ Meta k=6", AR1 + CT2 + ["MetaLog_l6"]), ("+ Meta k=9", AR1 + CT2 + ["MetaLog_l9"]),
                     ("+ dummies de mês + Meta k=6", AR1 + CT2 + MESES + ["MetaLog_l6"]),
                     ("+ dummies de mês + Meta k=9", AR1 + CT2 + MESES + ["MetaLog_l9"])):
        dd = recorte(p, INI_PADRAO, FIM).dropna(subset=["log_N_MA"] + xs)
        e = []
        for i in range(108, len(dd)):
            tr, te = dd.iloc[:i], dd.iloc[i:i + 1]
            mod = sm.OLS(tr["log_N_MA"], sm.add_constant(tr[xs])).fit()
            e.append(float(te["log_N_MA"].iloc[0] - mod.predict(sm.add_constant(te[xs], has_constant="add")).iloc[0]))
        oos.append({"modelo": nome, "RMSE": float(np.sqrt(np.mean(np.square(e)))), "MAE": float(np.mean(np.abs(e))),
                    "n_prev": len(e)})
    O = pd.DataFrame(oos)
    O["RMSE vs AR(1) (%)"] = 100 * (O["RMSE"] / O.loc[0, "RMSE"] - 1)
    O.to_csv(SAIDA / "avaliacao_fora_da_amostra.csv", index=False)
    print("\n== AVALIAÇÃO FORA DA AMOSTRA (Ibov e IBC-Br do próprio mês: dados já conhecidos) ==")
    print(O.round(4).to_string())

    # ---------- gráficos ----------
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    ax.errorbar(P["k"], P["coef"], yerr=[P["coef"] - P["ic95_inf"], P["ic95_sup"] - P["coef"]], fmt="s-",
                color="#1f3864", capsize=4)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(list(LAGS))
    ax.set_xlabel("Defasagem k (meses)")
    ax.set_ylabel("Coeficiente de Meta_log sobre ln(M&A)")
    ax.set_title("Efeito do nível da Selic sobre M&A, por defasagem (IC 95% HAC)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "coef_nivel_por_defasagem.png")
    plt.close(fig)

    x = p.loc[INICIO:FIM]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, dpi=150)
    ax[0].plot(x.index.to_timestamp(), x["N_MA"], color="#1f3864")
    ax[0].set_title("Nº mensal de anúncios de M&A (amostra filtrada)")
    ax[1].step(x.index.to_timestamp(), x["Meta"], color="#b7472a", where="post")
    ax[1].set_title("Meta Selic (SGS 432), fim do mês, % a.a.")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "series_ma_e_selic.png")
    plt.close(fig)

    for k in (K_PRINCIPAL, K_ALTERNATIVO):
        h = hacs[k]
        idx = h.fittedvalues.index.to_timestamp()
        res = h.resid
        fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), dpi=150)
        ax[0].plot(idx, h.model.endog, color="#555555", lw=1.2, label="ln(M&A) observado")
        ax[0].plot(idx, h.fittedvalues, color="#1f3864", lw=1.2, label=f"ajustado (k={k})")
        ax[0].set_title(f"Modelo k={k}: observado e ajustado")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.25)
        ax[1].plot(idx, res, color="#1f3864", lw=1)
        ax[1].axhline(0, color="k", lw=0.8)
        for sgn in (2, -2):
            ax[1].axhline(sgn * res.std(), color="#d62728", lw=0.8, ls="--")
        ax[1].set_title("Resíduos")
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(SAIDA / f"ajuste_e_residuos_k{k}.png")
        plt.close(fig)
        pd.DataFrame({"ajustado": h.fittedvalues, "residuo": res}).to_csv(SAIDA / f"residuos_k{k}.csv")

    print("\nPronto. Saída em", SAIDA)


if __name__ == "__main__":
    main()
