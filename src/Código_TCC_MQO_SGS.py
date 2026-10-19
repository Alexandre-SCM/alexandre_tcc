"""
A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
Evidências dos Ciclos da Selic (2005-2024)

MQO MENSAL com séries puxadas DIRETO do SGS/BCB (sem coletar nada à mão)
=========================================================================

O que este script faz de diferente das versões 1-4:

  1. SÉRIES VIA API (SGS/BCB), baixadas pelo próprio código e guardadas em ./dados_sgs
     (cache). Nenhuma série macro precisa ser copiada manualmente para a planilha:
        - 432   Meta Selic definida pelo Copom (diária)        -> VARIÁVEL DE POLÍTICA
        - 433   IPCA, variação mensal (%)                       -> controle
        - 24364 IBC-Br com ajuste sazonal (índice)              -> controle (atividade)
        - 3695  Câmbio R$/US$ (venda), fim de período           -> controle
        - 4189  Selic acumulada no mês, anualizada (%)          -> só checagem de consistência
     Se o download falhar (sem internet, BCB fora do ar), o código usa o cache; se não
     houver cache, usa a planilha como último recurso e AVISA, série por série, de onde
     veio cada dado (tabela "fontes_dados.csv" e impressão no console).
     Não vêm do SGS (continuam da planilha, são de outras fontes): Ibovespa (B3) e
     IIE-Br (FGV).

  2. VARIÁVEL DE POLÍTICA = META SELIC (SGS 432), no fim de cada mês, em variação mensal
     (ΔMeta, em p.p.). A meta só muda por decisão do Copom — não depende de nº de dias
     úteis (problema da série 4390 apontado em 06/10) e é a variável "de política" de fato.

  3. DEFASAGEM CORRETA: dMeta_l{k} = ΔMeta ocorrida k meses ANTES de t.  k=0 é a variação
     do próprio mês (contemporâneo), k=1 é a do mês anterior, etc. (corrige a convenção
     das versões 2-4, em que "lag 1" era a variação do próprio mês).

  4. DEFASAGENS EM DIFERENTES PERÍODOS:
        a) grade k = 0..12 meses, amostra completa;
        b) a mesma grade em SUBPERÍODOS (1ª metade, 2ª metade, sem Covid, pós-2016,
           ciclo pós-Covid);
        c) coeficiente acumulado (soma das defasagens 0..L) em janela móvel de 10 anos;
        d) modelos de defasagem distribuída (acumulado 0-3, 0-6 e 0-12 meses) com teste
           conjunto;
        e) correção de múltiplos testes (Holm) sobre a grade de 13 defasagens.

  5. Hipóteses do Projeto de Pesquisa:
        H1  Selic -> M&A negativo, com defasagem (1, 3 e 6 meses)         [testada]
        H2  incerteza (IIE-Br) importa                                     [controle de robustez]
        H3  assimetria (aperto x afrouxamento)                             [testada: ΔMeta+ e ΔMeta-]

  6. FILTROS DA BASE NO CÓDIGO (iguais aos das versões 1-4): Completed + alvo Brazil +
     Deal Size > US$ 0,5 MM + % adquirida > 50%. Nada de colunas "Filtro .../Sim".

  7. Diagnóstico: Breusch-Godfrey (no lugar do Durbin-Watson, inválido com dependente
     defasada), Breusch-Pagan, Jarque-Bera, VIF, ADF. Erros-padrão HAC (Newey-West).

Saídas em ./resultados_mqo_sgs  |  Cache das séries em ./dados_sgs
Não substitui as versões anteriores — fica lado a lado.
"""

from pathlib import Path
import json
import time
import warnings
import urllib.request
from datetime import date

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss, coint

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados_mqo_sgs"
CACHE = BASE / "dados_sgs"
SAIDA.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

ATUALIZAR_SGS = False        # True = baixa de novo mesmo que exista cache
SGS_INICIO_ANO = 2003        # baixa desde 2003 para ter defasagens já em 2005
SGS = {"meta_selic": 432, "ipca": 433, "ibc_br_sa": 24364, "cambio": 3695, "selic_4189": 4189}
CACHE_LEGADO_META = BASE / "dados_focus" / "selic_meta_sgs432.csv"   # baixado pelo VAR-surpresa

# --- filtros da base (iguais às versões 1-4) ---
DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0
REMOVER_DUPLICATAS = False
# duplicidades conferidas na fonte (linhas idênticas): mantém-se uma ocorrência de cada (2.292 -> 2.290)
IDS_DUPLICADOS = ["706879MM", "734331MM"]

INICIO, FIM = "2005-01", "2024-12"
MAX_LAG = 12
COVID_INICIO, COVID_FIM = "2020-03", "2020-12"
NW_LAGS = 6                   # Newey-West (mesmo das versões anteriores)
JANELA_MOVEL = 120            # 10 anos

CTRL_BASE = ["Ibov_Ret", "dIBC_Br", "Covid"]
CTRL_ROBUSTO = CTRL_BASE + ["IPCA", "dCambio", "dIIE_Br"]

SUBPERIODOS = {
    "Completo": (None, "2024-12"),
    "1ª metade (até 2014)": (None, "2014-12"),
    "2ª metade (2015-2024)": ("2015-01", "2024-12"),
    "Sem Covid (até 2019)": (None, "2019-12"),
    "Pós-2016 (2016-2024)": ("2016-01", "2024-12"),
    "Ciclo pós-Covid (2021-2024)": ("2021-01", "2024-12"),
}

FONTES: dict = {}


# =============================================================================
# 1. SGS/BCB — download com cache e fallback explícito
# =============================================================================


def _get_json(url: str, tentativas: int = 2, timeout: int = 60):
    ultimo = None
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TCC Ibmec-DF)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"falha ao baixar {url}: {ultimo}")


def baixar_sgs(codigo: int) -> pd.Series:
    """Baixa a série em blocos de 5 anos (a API limita o intervalo de séries diárias)."""
    partes = []
    ano = SGS_INICIO_ANO
    hoje = date.today()
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


def _ler_cache(arq: Path) -> pd.Series:
    df = pd.read_csv(arq)
    df.columns = ["data", "valor"] + list(df.columns[2:])
    df["data"] = pd.to_datetime(df["data"])
    return pd.Series(pd.to_numeric(df["valor"], errors="coerce").values,
                     index=pd.DatetimeIndex(df["data"])).dropna().sort_index()


def carregar_sgs(nome: str, codigo: int, fallback=None):
    """Ordem: (1) download SGS [se ATUALIZAR_SGS ou sem cache]  (2) cache  (3) cache legado
    (só meta) (4) fallback da planilha. A fonte efetivamente usada fica registrada."""
    arq = CACHE / f"sgs_{codigo}.csv"
    erro = ""
    if ATUALIZAR_SGS or not arq.exists():
        try:
            s = baixar_sgs(codigo)
            s.rename("valor").rename_axis("data").to_csv(arq)
            FONTES[nome] = (f"SGS {codigo}", "download ao vivo (API BCB)", s.index.min(), s.index.max(), len(s))
            return s
        except Exception as e:  # noqa: BLE001
            erro = str(e)[:120]
    if arq.exists():
        s = _ler_cache(arq)
        FONTES[nome] = (f"SGS {codigo}", "cache local ./dados_sgs" + (f" (download falhou: {erro})" if erro else ""),
                        s.index.min(), s.index.max(), len(s))
        return s
    if nome == "meta_selic" and CACHE_LEGADO_META.exists():
        s = _ler_cache(CACHE_LEGADO_META)
        FONTES[nome] = (f"SGS {codigo}", f"cache legado dados_focus (download falhou: {erro})",
                        s.index.min(), s.index.max(), len(s))
        return s
    if fallback is not None:
        s, desc = fallback()
        FONTES[nome] = (f"SGS {codigo}", f"*** FALLBACK PLANILHA: {desc} *** (download falhou: {erro})",
                        s.index.min(), s.index.max(), len(s))
        print(f"  [AVISO] {nome} (SGS {codigo}) NÃO veio do SGS: {desc}. Motivo: {erro}")
        return s
    raise RuntimeError(f"Sem como obter {nome} (SGS {codigo}). Erro: {erro}")


def _mensal(s: pd.Series, como: str = "last") -> pd.Series:
    """Série (diária ou mensal) -> série mensal indexada por Period('M')."""
    g = s.groupby(s.index.to_period("M"))
    out = g.last() if como == "last" else g.mean()
    idx = pd.period_range(out.index.min(), "2024-12", freq="M")
    return out.reindex(idx).ffill()


# =============================================================================
# 2. BASE DE NEGÓCIOS (filtros no código) e planilha (Ibovespa, IIE-Br e fallbacks)
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
    # remove só a 2ª ocorrência dos IDs conferidos, e apenas se a linha for idêntica à 1ª
    rep = out["Deal ID"].isin(IDS_DUPLICADOS) & out.duplicated(keep="first")
    out = out[~rep]
    if dedup:
        out = out.drop_duplicates(["Company Name (Target/Issuer)", "Announcement Date", "valor"])
    return out


def contagem_mensal(negocios: pd.DataFrame) -> pd.Series:
    per = negocios["Announcement Date"].dt.to_period("M")
    s = per.value_counts().sort_index()
    idx = pd.period_range(INICIO, FIM, freq="M")
    return s.reindex(idx, fill_value=0).rename("N_MA")


def _serie_planilha(m, col_data, col_val, nome):
    s = m.iloc[:, [col_data, col_val]].copy()
    s.columns = ["data", nome]
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    s = s.dropna()
    s.index = s["data"].dt.to_period("M")
    return pd.to_numeric(s[nome], errors="coerce")


def _planilha_macro():
    return pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)


def montar_macro() -> pd.DataFrame:
    m = _planilha_macro()

    def fb_ipca():
        s = _serie_planilha(m, 7, 8, "IPCA")
        return s.set_axis(s.index.to_timestamp()), "IPCA mensal da planilha (cópia do SGS 433)"

    def fb_ibc():
        s = _serie_planilha(m, 13, 14, "IBC")
        return (s.set_axis(s.index.to_timestamp()),
                "IBC-Br da planilha = SGS 24363 SEM ajuste sazonal (o código usa 24364 com ajuste)")

    meta = carregar_sgs("meta_selic", SGS["meta_selic"])
    ipca = carregar_sgs("ipca", SGS["ipca"], fallback=fb_ipca)
    ibc = carregar_sgs("ibc_br_sa", SGS["ibc_br_sa"], fallback=fb_ibc)
    try:
        cambio = carregar_sgs("cambio", SGS["cambio"])
    except RuntimeError:
        cambio = None
        FONTES["cambio"] = ("SGS 3695", "INDISPONÍVEL (sem download nem cache) — controle dCambio omitido",
                            pd.NaT, pd.NaT, 0)
        print("  [AVISO] câmbio (SGS 3695) indisponível — dCambio ficará fora dos controles robustos.")
    try:
        s4189 = carregar_sgs("selic_4189", SGS["selic_4189"])
    except RuntimeError:
        s4189 = None

    macro = pd.concat({
        "Meta": _mensal(meta, "last"),
        "Meta_media": _mensal(meta, "mean"),
        "IPCA": _mensal(ipca, "last"),
        "IBC_Br": _mensal(ibc, "last"),
    }, axis=1)
    if cambio is not None:
        macro["Cambio"] = _mensal(cambio, "last")
    if s4189 is not None:
        macro["Selic4189"] = _mensal(s4189, "last")

    # --- fontes que NÃO são do SGS (planilha): Ibovespa (B3) e IIE-Br (FGV) ---
    ibov = pd.concat([_serie_planilha(m, 28, 29, "Ibov_Nivel"), _serie_planilha(m, 28, 30, "Ibov_Vol")], axis=1)
    iie = _serie_planilha(m, 19, 20, "IIE_Br")
    FONTES["ibovespa"] = ("B3 (planilha)", "planilha Macro_Controls — não é SGS", ibov.index.min().to_timestamp(),
                          ibov.index.max().to_timestamp(), len(ibov))
    FONTES["iie_br"] = ("FGV (planilha)", "planilha Macro_Controls — não é SGS", iie.index.min().to_timestamp(),
                        iie.index.max().to_timestamp(), len(iie))
    macro = macro.join(ibov, how="outer").join(iie.rename("IIE_Br"), how="outer")
    return macro


def montar_painel(negocios: pd.DataFrame) -> pd.DataFrame:
    macro = montar_macro()
    idx = pd.period_range("2003-01", FIM, freq="M")
    p = macro.reindex(idx)
    p["N_MA"] = contagem_mensal(negocios).reindex(idx)
    p["log_N_MA"] = np.log(p["N_MA"])
    p["log_N_MA_lag1"] = p["log_N_MA"].shift(1)

    # política monetária: variação mensal da META (p.p.), parte positiva (aperto) e negativa (corte)
    p["dMeta"] = p["Meta"].diff()
    p["dMeta_pos"] = p["dMeta"].clip(lower=0)
    p["dMeta_neg"] = p["dMeta"].clip(upper=0)
    for k in range(MAX_LAG + 1):
        p[f"dMeta_l{k}"] = p["dMeta"].shift(k)           # k=0: mês corrente; k=1: mês anterior...
        p[f"dMeta_pos_l{k}"] = p["dMeta_pos"].shift(k)
        p[f"dMeta_neg_l{k}"] = p["dMeta_neg"].shift(k)
        p[f"Meta_l{k}"] = p["Meta"].shift(k)

    # controles
    p["Ibov_Ret"] = 100 * np.log(p["Ibov_Nivel"]).diff()
    p["dIBC_Br"] = p["IBC_Br"].pct_change() * 100
    if "Cambio" in p:
        p["dCambio"] = p["Cambio"].pct_change() * 100
    p["dIIE_Br"] = p["IIE_Br"].diff()
    s = p.index.astype(str)
    p["Covid"] = ((s >= COVID_INICIO) & (s <= COVID_FIM)).astype(int)
    p["tendencia"] = np.arange(len(p)) / 100.0
    p.index.name = "Periodo"
    return p


def checagem_consistencia(p: pd.DataFrame):
    """Meta (432) x Selic acumulada anualizada (4189): devem andar juntas."""
    if "Selic4189" not in p:
        print("  (SGS 4189 indisponível nesta execução — checagem de consistência pulada)")
        return None
    d = p[["Meta_media", "Selic4189"]].dropna()
    corr = d.corr().iloc[0, 1]
    dif = (d["Selic4189"] - d["Meta_media"]).abs().max()
    print(f"  Consistência Meta (média do mês) x Selic 4189: correlação={corr:.4f}, "
          f"maior diferença absoluta={dif:.2f} p.p.")
    return {"correlacao": corr, "dif_max_pp": dif}


# =============================================================================
# 3. ESTIMAÇÃO — utilidades
# =============================================================================


def ols_hac(y: pd.Series, X: pd.DataFrame, nw: int = NW_LAGS):
    X = sm.add_constant(X, has_constant="add")
    manter = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    X = X[manter]
    plain = sm.OLS(y, X, missing="drop").fit()
    hac = sm.OLS(y, X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": nw, "use_correction": True}, use_t=True)
    return hac, plain


def _f(x):
    return float(np.squeeze(x))


def _linha_coef(hac, var, **extra):
    ic = hac.conf_int().loc[var]
    return {"coef": hac.params[var], "ep_HAC": hac.bse[var], "t": hac.tvalues[var],
            "p_valor": hac.pvalues[var], "ic95_inf": ic.iloc[0], "ic95_sup": ic.iloc[1],
            "n": int(hac.nobs), "R2_aj": hac.rsquared_adj, "BIC": hac.bic, **extra}


def recorte(p: pd.DataFrame, ini, fim) -> pd.DataFrame:
    return p.loc[ini:fim] if ini else p.loc[:fim]


def inicio_comum(p: pd.DataFrame, controles) -> pd.Period:
    cols = ["log_N_MA", "log_N_MA_lag1"] + [f"dMeta_l{k}" for k in range(MAX_LAG + 1)] + list(controles)
    return p.dropna(subset=cols).index.min()


def controles_disponiveis(p, lista):
    return [c for c in lista if c in p.columns and p[c].notna().sum() > 50]


# =============================================================================
# 4. ESTACIONARIEDADE
# =============================================================================


def testes_raiz_unitaria(p: pd.DataFrame) -> pd.DataFrame:
    cand = {"log(N_MA)": "log_N_MA", "Meta Selic (nível)": "Meta", "ΔMeta Selic": "dMeta",
            "Ibov_Ret": "Ibov_Ret", "IBC-Br SA (nível)": "IBC_Br", "ΔIBC-Br SA (%)": "dIBC_Br",
            "IPCA": "IPCA", "ΔCâmbio (%)": "dCambio", "IIE-Br (nível)": "IIE_Br", "ΔIIE-Br": "dIIE_Br"}
    linhas = []
    for nome, col in cand.items():
        if col not in p:
            continue
        s = p.loc[INICIO:FIM, col].dropna()
        if len(s) < 30:
            continue
        a = adfuller(s, regression="c", autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
        linhas.append({"Variável": nome, "ADF stat": a[0], "ADF p-valor": a[1], "ADF: estacionária a 5%": a[1] < .05,
                       "KPSS stat": k[0], "KPSS p-valor (trunc. 0,01-0,10)": k[1],
                       "KPSS: não rejeita estacionariedade": k[1] >= .05})
    return pd.DataFrame(linhas)


# =============================================================================
# 5. GRADE DE DEFASAGENS (k = 0..12), por período
# =============================================================================


def grade(p: pd.DataFrame, ini, fim, controles, rotulo="") -> pd.DataFrame:
    d = recorte(p, ini, fim)
    linhas = []
    for k in range(MAX_LAG + 1):
        col = f"dMeta_l{k}"
        cols = ["log_N_MA", "log_N_MA_lag1", col] + list(controles)
        dd = d.dropna(subset=cols)
        if len(dd) < 30:
            continue
        hac, _ = ols_hac(dd["log_N_MA"], dd[["log_N_MA_lag1", col] + list(controles)])
        linhas.append({"periodo": rotulo, "k_meses": k, **_linha_coef(hac, col)})
    return pd.DataFrame(linhas)


def grade_todos_periodos(p, controles):
    ini_c = inicio_comum(p, controles)
    partes = []
    for nome, (ini, fim) in SUBPERIODOS.items():
        partes.append(grade(p, ini or ini_c, fim, controles, nome))
    return pd.concat(partes, ignore_index=True)


def holm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_Holm"] = multipletests(df["p_valor"], method="holm")[1]
    return df


# =============================================================================
# 6. DEFASAGEM DISTRIBUÍDA (efeito acumulado) e ASSIMETRIA
# =============================================================================


def bloco_distribuido(p, L, controles, ini=None, fim=FIM, rotulo=""):
    cols_l = [f"dMeta_l{k}" for k in range(L + 1)]
    d = recorte(p, ini or inicio_comum(p, controles), fim).dropna(
        subset=["log_N_MA", "log_N_MA_lag1"] + cols_l + list(controles))
    hac, plain = ols_hac(d["log_N_MA"], d[["log_N_MA_lag1"] + cols_l + list(controles)])
    tt = hac.t_test(" + ".join(cols_l) + " = 0")
    w = hac.wald_test(", ".join(f"{c} = 0" for c in cols_l), use_f=True, scalar=True)
    ci = np.squeeze(tt.conf_int())
    return {"modelo": rotulo or f"acumulado 0-{L}", "L": L, "soma_coef": _f(tt.effect), "ep_HAC": _f(tt.sd),
            "p_soma": _f(tt.pvalue), "ic95_inf": float(ci[0]), "ic95_sup": float(ci[1]),
            "p_conjunto(todas=0)": _f(w.pvalue), "coef_log_N_MA_lag1": hac.params["log_N_MA_lag1"],
            "n": int(hac.nobs), "R2_aj": hac.rsquared_adj, "BIC": hac.bic}, plain


def tabela_distribuida(p):
    linhas, plains = [], {}
    for rot, ctrl in (("controles base", controles_disponiveis(p, CTRL_BASE)),
                      ("controles robustos (+" + ", ".join(c for c in controles_disponiveis(p, CTRL_ROBUSTO)
                                                  if c not in CTRL_BASE) + ")", controles_disponiveis(p, CTRL_ROBUSTO))):
        for L in (3, 6, 12):
            linha, plain = bloco_distribuido(p, L, ctrl, rotulo=f"acumulado 0-{L} | {rot}")
            linha["controles"] = rot
            linhas.append(linha)
            plains[linha["modelo"]] = plain
    return pd.DataFrame(linhas), plains


def tabela_assimetria(p, controles):
    ini = inicio_comum(p, controles)
    linhas = []
    for k in (0, 1, 2, 3, 6):
        cp, cn = f"dMeta_pos_l{k}", f"dMeta_neg_l{k}"
        d = recorte(p, ini, FIM).dropna(subset=["log_N_MA", "log_N_MA_lag1", cp, cn] + list(controles))
        hac, _ = ols_hac(d["log_N_MA"], d[["log_N_MA_lag1", cp, cn] + list(controles)])
        w = hac.t_test(f"{cp} - {cn} = 0")
        linhas.append({"modelo": f"lag {k}", "coef_aperto(ΔMeta+)": hac.params[cp], "p_aperto": hac.pvalues[cp],
                       "coef_corte(ΔMeta-)": hac.params[cn], "p_corte": hac.pvalues[cn],
                       "p(aperto = corte)": _f(w.pvalue), "n": int(hac.nobs)})
    for L in (3, 6):
        cps = [f"dMeta_pos_l{k}" for k in range(L + 1)]
        cns = [f"dMeta_neg_l{k}" for k in range(L + 1)]
        d = recorte(p, ini, FIM).dropna(subset=["log_N_MA", "log_N_MA_lag1"] + cps + cns + list(controles))
        hac, _ = ols_hac(d["log_N_MA"], d[["log_N_MA_lag1"] + cps + cns + list(controles)])
        tp, tn = hac.t_test(" + ".join(cps) + " = 0"), hac.t_test(" + ".join(cns) + " = 0")
        td = hac.t_test(" + ".join(cps) + " - (" + " + ".join(cns) + ") = 0")
        linhas.append({"modelo": f"acumulado 0-{L}", "coef_aperto(ΔMeta+)": _f(tp.effect), "p_aperto": _f(tp.pvalue),
                       "coef_corte(ΔMeta-)": _f(tn.effect), "p_corte": _f(tn.pvalue),
                       "p(aperto = corte)": _f(td.pvalue), "n": int(hac.nobs)})
    return pd.DataFrame(linhas)


def tabela_nivel(p, controles):
    """H1 'literal' (nível da Meta). Com e sem tendência — o nível costuma gerar regressão espúria."""
    ini = inicio_comum(p, controles)
    linhas = []
    for k in (1, 3, 6, 12):
        for com_tend in (False, True):
            col = f"Meta_l{k}"
            xs = ["log_N_MA_lag1", col] + list(controles) + (["tendencia"] if com_tend else [])
            d = recorte(p, ini, FIM).dropna(subset=["log_N_MA"] + xs)
            hac, _ = ols_hac(d["log_N_MA"], d[xs])
            linhas.append({"k_meses": k, "com_tendencia_linear": com_tend, **_linha_coef(hac, col)})
    return pd.DataFrame(linhas)


def tabela_nivel_e_variacao(p, controles):
    """Nível da Meta (efeito 'de estoque') e variação contemporânea juntos — tipo correção de erro.
    Separa o efeito de longo prazo (nível) do de curto prazo (variação)."""
    ini = inicio_comum(p, controles)
    linhas = []
    for k in (1, 3, 6, 12):
        xs = ["log_N_MA_lag1", f"Meta_l{k}", "dMeta_l0"] + list(controles)
        d = recorte(p, ini, FIM).dropna(subset=["log_N_MA"] + xs)
        hac, _ = ols_hac(d["log_N_MA"], d[xs])
        linhas.append({"k_nivel": k, "coef_nivel": hac.params[f"Meta_l{k}"], "p_nivel": hac.pvalues[f"Meta_l{k}"],
                       "coef_dMeta_l0": hac.params["dMeta_l0"], "p_dMeta_l0": hac.pvalues["dMeta_l0"],
                       "n": int(hac.nobs)})
    return pd.DataFrame(linhas)


def _estimar_sem(p, controles, anos_fora, L=3):
    """Acumulado 0-L e nível (k=3) excluindo meses dos anos em 'anos_fora'."""
    cols_l = [f"dMeta_l{k}" for k in range(L + 1)]
    d = recorte(p, inicio_comum(p, controles), FIM)
    d = d[~d.index.year.isin(anos_fora)].dropna(subset=["log_N_MA", "log_N_MA_lag1"] + cols_l + list(controles))
    hac, _ = ols_hac(d["log_N_MA"], d[["log_N_MA_lag1"] + cols_l + list(controles)])
    tt = hac.t_test(" + ".join(cols_l) + " = 0")
    xs = ["log_N_MA_lag1", "Meta_l3"] + list(controles)
    dn = recorte(p, inicio_comum(p, controles), FIM)
    dn = dn[~dn.index.year.isin(anos_fora)].dropna(subset=["log_N_MA"] + xs)
    hn, _ = ols_hac(dn["log_N_MA"], dn[xs])
    return {"soma_0_3": _f(tt.effect), "ep": _f(tt.sd), "p_soma_0_3": _f(tt.pvalue),
            "coef_nivel_k3": hn.params["Meta_l3"], "p_nivel_k3": hn.pvalues["Meta_l3"], "n": int(hac.nobs)}


def influencia_episodios(p, controles):
    linhas = [{"excluido": "(nenhum)", **_estimar_sem(p, controles, [])}]
    for rot, anos in (("2020-2022", [2020, 2021, 2022]), ("2020", [2020]), ("2021", [2021]),
                      ("2021-2022", [2021, 2022])):
        linhas.append({"excluido": rot, **_estimar_sem(p, controles, anos)})
    for a in range(2006, 2025):
        linhas.append({"excluido": f"só {a}", **_estimar_sem(p, controles, [a])})
    return pd.DataFrame(linhas)


def cointegracao(p):
    """Engle-Granger entre log(M&A) e o NÍVEL da Meta: se cointegram, a relação em nível não é espúria."""
    d = p.loc[INICIO:FIM, ["log_N_MA", "Meta"]].dropna()
    t, pv, crit = coint(d["log_N_MA"], d["Meta"], trend="c", autolag="aic")
    return {"Engle-Granger t": t, "p_valor": pv, "crítico 1%": crit[0], "crítico 5%": crit[1],
            "crítico 10%": crit[2], "cointegram a 5%": bool(pv < .05), "n": len(d)}


def tabela_controles(p):
    """H2 (incerteza, IIE-Br) e demais controles no modelo acumulado 0-3 com controles robustos."""
    ctrl = controles_disponiveis(p, CTRL_ROBUSTO)
    cols_l = [f"dMeta_l{k}" for k in range(4)]
    linhas = []
    for rot, cs in (("IIE-Br em variação", ctrl),
                    ("IIE-Br em nível", ["IIE_Br" if c == "dIIE_Br" else c for c in ctrl])):
        d = recorte(p, inicio_comum(p, cs), FIM).dropna(subset=["log_N_MA", "log_N_MA_lag1"] + cols_l + cs)
        hac, _ = ols_hac(d["log_N_MA"], d[["log_N_MA_lag1"] + cols_l + cs])
        for v in ["log_N_MA_lag1"] + cs:
            linhas.append({"especificação": rot, "variável": v, "coef": hac.params[v], "ep_HAC": hac.bse[v],
                           "p_valor": hac.pvalues[v], "n": int(hac.nobs)})
    return pd.DataFrame(linhas)


# =============================================================================
# 7. JANELA MÓVEL (efeito acumulado 0-3 em janelas de 10 anos)
# =============================================================================


def janela_movel(p, controles, L=3):
    cols_l = [f"dMeta_l{k}" for k in range(L + 1)]
    d = recorte(p, inicio_comum(p, controles), FIM).dropna(
        subset=["log_N_MA", "log_N_MA_lag1"] + cols_l + list(controles))
    linhas = []
    for i in range(0, len(d) - JANELA_MOVEL + 1):
        w = d.iloc[i:i + JANELA_MOVEL]
        hac, _ = ols_hac(w["log_N_MA"], w[["log_N_MA_lag1"] + cols_l + list(controles)])
        tt = hac.t_test(" + ".join(cols_l) + " = 0")
        linhas.append({"fim_janela": w.index[-1].to_timestamp(), "soma_coef": _f(tt.effect), "ep": _f(tt.sd),
                       "p": _f(tt.pvalue)})
    return pd.DataFrame(linhas)


# =============================================================================
# 8. DIAGNÓSTICOS
# =============================================================================


def diagnosticos(plain, nome):
    r = plain.resid
    X = plain.model.exog
    vif = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    return {"modelo": nome, "n": int(plain.nobs), "R2_aj": plain.rsquared_adj,
            "Breusch-Godfrey p (6)": acorr_breusch_godfrey(plain, nlags=6)[1],
            "Breusch-Godfrey p (12)": acorr_breusch_godfrey(plain, nlags=12)[1],
            "Breusch-Pagan p": het_breuschpagan(r, X)[1], "Jarque-Bera p": jarque_bera(r)[1],
            "VIF máx": max(vif) if vif else np.nan}


# =============================================================================
# 9. GRÁFICOS
# =============================================================================


def graf_grade(g_base, g_rob):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for g, cor, rot, off in ((g_base, "#1f4e79", "controles base", -0.12), (g_rob, "#b7472a", "controles robustos", 0.12)):
        if g is None or g.empty:
            continue
        x = g["k_meses"] + off
        ax.errorbar(x, g["coef"], yerr=[g["coef"] - g["ic95_inf"], g["ic95_sup"] - g["coef"]],
                    fmt="o", color=cor, capsize=3, label=rot)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Defasagem k (meses) — ΔMeta ocorrida k meses antes")
    ax.set_ylabel("Coeficiente sobre log(M&A) (≈ % por +1 p.p.)")
    ax.set_title("Efeito de ΔMeta Selic sobre log(M&A), por defasagem — IC 95% (HAC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "grade_defasagens.png", dpi=200)
    plt.close(fig)


def graf_heatmap(g):
    piv = g.pivot(index="periodo", columns="k_meses", values="t").reindex(list(SUBPERIODOS))
    pv = g.pivot(index="periodo", columns="k_meses", values="p_valor").reindex(list(SUBPERIODOS))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, pvl = piv.values[i, j], pv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.1f}" + ("*" if pvl < .05 else ""), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="estatística t (HAC); * = p<0,05")
    ax.set_xlabel("Defasagem k (meses)")
    ax.set_title("Estatística t do coeficiente de ΔMeta por período e defasagem")
    fig.tight_layout()
    fig.savefig(SAIDA / "heatmap_periodos_defasagens.png", dpi=200)
    plt.close(fig)


def graf_movel(jm):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(jm["fim_janela"], jm["soma_coef"], color="#1f4e79")
    ax.fill_between(jm["fim_janela"], jm["soma_coef"] - 1.96 * jm["ep"], jm["soma_coef"] + 1.96 * jm["ep"],
                    color="#1f4e79", alpha=.2)
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Efeito acumulado (ΔMeta 0 a 3 meses) em janelas móveis de 10 anos — IC 95%")
    ax.set_xlabel("Fim da janela")
    fig.tight_layout()
    fig.savefig(SAIDA / "janela_movel_acumulado.png", dpi=200)
    plt.close(fig)


def graf_series(p):
    d = p.loc[INICIO:FIM]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(d.index.to_timestamp(), d["N_MA"], color="#1f4e79")
    ax[0].set_title("Nº mensal de anúncios de M&A (amostra filtrada)")
    ax[1].step(d.index.to_timestamp(), d["Meta"], color="#b7472a", where="post")
    ax[1].set_title("Meta Selic (SGS 432), fim do mês, % a.a.")
    fig.tight_layout()
    fig.savefig(SAIDA / "series_mqo_sgs.png", dpi=200)
    plt.close(fig)


# =============================================================================
# 10. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 74)
    print("1. DADOS")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final (filtros no código): {len(negocios):,}")
    painel = montar_painel(negocios)

    fontes = pd.DataFrame(FONTES, index=["série", "origem usada", "início", "fim", "n obs."]).T
    fontes.to_csv(SAIDA / "fontes_dados.csv")
    print("\nDe onde veio cada série:")
    print(fontes.to_string())
    checagem_consistencia(painel)

    ctrl_base = controles_disponiveis(painel, CTRL_BASE)
    ctrl_rob = controles_disponiveis(painel, CTRL_ROBUSTO)
    ini_c = inicio_comum(painel, ctrl_base)
    print(f"\nAmostra comum para a grade de defasagens (0..{MAX_LAG}): {ini_c} a {FIM} "
          f"(n={len(painel.loc[ini_c:FIM])})")
    painel.loc[INICIO:FIM].to_csv(SAIDA / "painel_mensal_mqo_sgs.csv")
    graf_series(painel)

    print("\n2. RAIZ UNITÁRIA (ADF)")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria.csv", index=False)
    print(ru.round(4).to_string(index=False))

    print("\n3. GRADE DE DEFASAGENS k=0..12 — amostra completa (ΔMeta_{t-k}), HAC")
    g_base = holm(grade(painel, ini_c, FIM, ctrl_base, "Completo | controles base"))
    g_rob = holm(grade(painel, inicio_comum(painel, ctrl_rob), FIM, ctrl_rob, "Completo | controles robustos"))
    pd.concat([g_base, g_rob]).to_csv(SAIDA / "grade_defasagens_completo.csv", index=False)
    cols_show = ["k_meses", "coef", "ep_HAC", "p_valor", "p_Holm", "ic95_inf", "ic95_sup", "n"]
    print("Controles base (Ibov_Ret, ΔIBC-Br, Covid):")
    print(g_base[cols_show].round(4).to_string(index=False))
    print("Controles robustos (+ " + ", ".join(c for c in ctrl_rob if c not in ctrl_base) + "):")
    print(g_rob[cols_show].round(4).to_string(index=False))
    graf_grade(g_base, g_rob)

    print("\n4. A MESMA GRADE EM DIFERENTES PERÍODOS (controles base)")
    g_per = grade_todos_periodos(painel, ctrl_base)
    g_per.to_csv(SAIDA / "grade_subperiodos.csv", index=False)
    resumo = g_per.assign(sig=g_per["p_valor"] < .05).groupby("periodo", sort=False).agg(
        n=("n", "first"), defas_sig_5pct=("sig", "sum"),
        sinal_negativo=("coef", lambda s: int((s < 0).sum())), de=("k_meses", "count"))
    print(resumo.to_string())
    for nome in SUBPERIODOS:
        sub = g_per[(g_per["periodo"] == nome) & (g_per["p_valor"] < .05)]
        if len(sub):
            print(f"  {nome}: significativos a 5% em k={list(sub['k_meses'])} com coef. "
                  f"{[round(c, 3) for c in sub['coef']]}")
    graf_heatmap(g_per)

    print("\n5. DEFASAGEM DISTRIBUÍDA (efeito ACUMULADO = soma dos coeficientes de ΔMeta, t..t-L)")
    dist, plains = tabela_distribuida(painel)
    dist.to_csv(SAIDA / "defasagem_distribuida.csv", index=False)
    print(dist.drop(columns=["controles"]).round(4).to_string(index=False))

    print("\n6. ASSIMETRIA (H3): aperto (ΔMeta>0) x corte (ΔMeta<0)")
    assim = tabela_assimetria(painel, ctrl_base)
    assim.to_csv(SAIDA / "assimetria.csv", index=False)
    print(assim.round(4).to_string(index=False))

    print("\n7. MODELO EM NÍVEL (H1 literal: nível da Meta) — com e sem tendência linear")
    niv = tabela_nivel(painel, ctrl_base)
    niv.to_csv(SAIDA / "modelo_nivel.csv", index=False)
    print(niv[["k_meses", "com_tendencia_linear", "coef", "p_valor", "n"]].round(4).to_string(index=False))

    print("\n7b. NÍVEL + VARIAÇÃO NO MESMO MODELO (separa efeito de estoque x efeito de decisão)")
    nv = tabela_nivel_e_variacao(painel, ctrl_base)
    nv.to_csv(SAIDA / "nivel_e_variacao.csv", index=False)
    print(nv.round(4).to_string(index=False))

    print("\n7c. INFLUÊNCIA DE EPISÓDIOS/ANOS (acumulado ΔMeta 0-3 e nível k=3) — excluindo períodos")
    infl = influencia_episodios(painel, ctrl_base)
    infl.to_csv(SAIDA / "influencia_episodios.csv", index=False)
    print(infl.round(4).to_string(index=False))

    print("\n7d. COINTEGRAÇÃO (Engle-Granger): log(M&A) x nível da Meta — a relação em nível é espúria?")
    ci = pd.DataFrame([cointegracao(painel)])
    ci.to_csv(SAIDA / "cointegracao.csv", index=False)
    print(ci.round(4).to_string(index=False))

    print("\n7e. H2 (incerteza IIE-Br) e demais controles — modelo acumulado ΔMeta 0-3, controles robustos")
    tc = tabela_controles(painel)
    tc.to_csv(SAIDA / "controles_h2.csv", index=False)
    print(tc.round(4).to_string(index=False))

    print("\n8. JANELA MÓVEL DE 10 ANOS — efeito acumulado ΔMeta 0-3")
    jm = janela_movel(painel, ctrl_base)
    jm.to_csv(SAIDA / "janela_movel.csv", index=False)
    graf_movel(jm)
    sig = (jm["p"] < .05).mean() * 100
    print(f"Janelas: {len(jm)} | soma média={jm['soma_coef'].mean():.3f} | mín={jm['soma_coef'].min():.3f} | "
          f"máx={jm['soma_coef'].max():.3f} | % das janelas com p<0,05: {sig:.0f}%")

    print("\n9. DIAGNÓSTICOS dos modelos acumulados")
    diag = pd.DataFrame([diagnosticos(pl, nm) for nm, pl in plains.items()])
    diag.to_csv(SAIDA / "diagnosticos.csv", index=False)
    print(diag.round(4).to_string(index=False))

    print("\n10. LEITURA RÁPIDA (fatos, sem interpretação)")
    s_raw = g_base[g_base["p_valor"] < .05]
    s_holm = g_base[g_base["p_Holm"] < .05]
    print(f"- Grade completa (controles base): {len(s_raw)} de {len(g_base)} defasagens com p<0,05; "
          f"após correção de Holm: {len(s_holm)}.")
    print(f"- Sinais negativos (H1): {(g_base['coef'] < 0).sum()} de {len(g_base)}; "
          f"negativos e significativos a 5%: {((g_base['coef'] < 0) & (g_base['p_valor'] < .05)).sum()}.")
    print(f"- Defasagens do Projeto de Pesquisa (1, 3, 6): "
          + "; ".join(f"k={k}: coef={g_base.loc[g_base.k_meses == k, 'coef'].iloc[0]:+.3f} "
                      f"(p={g_base.loc[g_base.k_meses == k, 'p_valor'].iloc[0]:.3f})" for k in (1, 3, 6)))
    print("\nArquivos em:", SAIDA)


if __name__ == "__main__":
    main()
