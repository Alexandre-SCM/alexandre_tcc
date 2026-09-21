"""
Trabalho de Conclusão de Curso - IBMEC
Autor: Alexandre de Sousa Campos Menezes
"Título: A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
 Evidências dos Ciclos da Selic (2005-2024)"

Desenho empírico: VAR mensal agregado, replicando Batista, Lamounier, Mário e
Ferreira (2022, Contextus), com IRF de Cholesky + bootstrap.

    y_t = [N_MA, IIE_Br, IBC_Br, Ibov_Ret, Ibov_Vol, Selic]'

Hipóteses
    H1: choque de alta da Selic reduz o volume de M&A.
    H2: a incerteza (IIE-Br) tem efeito próprio sobre M&A, controlando pela Selic.
    H3: efeito assimétrico entre ciclos de alta e de baixa da Selic.

"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # salva figuras em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO  (todo critério de amostra/modelo fica aqui, explícito)
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados"
SAIDA.mkdir(exist_ok=True)

# Critérios de amostra (aplicados sobre as colunas NUMÉRICAS brutas)
DEAL_MIN_USD_MM = 0.5      # Deal Size (USD, MM) > 0,5
PCT_MIN = 50.0             # % of Target Shares Sought > 50 (controle)
REMOVER_DUPLICATAS = False  # 11 linhas suspeitas; ver secção 1 (decisão em aberto)

# Janela da amostra
INICIO, FIM = "2005-01", "2024-12"

# Especificação do VAR (ordem = ordem de Cholesky, como em Batista et al. 2022)
VARIAVEIS = ["N_MA", "IIE_Br", "IBC_Br", "Ibov_Ret", "Ibov_Vol", "Selic"]
MAXLAGS = 12
LAGS_PRINCIPAL = None      # None = escolhe por SBIC (parcimônia, como o paper)
HORIZONTE = 8              # meses
N_BOOT = 1000
NIVEL_IC = 90              # % da banda de confiança do bootstrap
SEMENTE = 42
CHOQUE_SELIC_PP = 1.0      # tamanho do choque de política (p.p.)

# "niveis" replica o paper; "diferencas" diferencia as séries que forem I(1)
TRANSFORMACAO = "niveis"

# =============================================================================
# 1. DADOS: leitura, filtros de amostra e painel mensal
# =============================================================================


def carregar_negocios() -> pd.DataFrame:
    """Lê a base bruta FactSet e aplica os critérios de amostra."""
    df = pd.read_excel(ARQUIVO, sheet_name="Transactions_Results", header=2)
    df["Announcement Date"] = pd.to_datetime(df["Announcement Date"])
    # IMPORTANTE: não usar as colunas 'Filtro ...' (Sim/-) do Excel como máscara.
    # A coluna de % trata texto como > 50 e infla a amostra (6.571 em vez de 2.292).
    df["valor"] = pd.to_numeric(df["Deal Size (USD, MM)"], errors="coerce")
    df["pct"] = pd.to_numeric(df["% of Target Shares Sought"], errors="coerce")
    return df


def aplicar_filtros(df: pd.DataFrame, deal_min=DEAL_MIN_USD_MM, pct_min=PCT_MIN,
                    dedup=REMOVER_DUPLICATAS) -> pd.DataFrame:
    m = (df["Deal Status"] == "Completed") & (df["Country (Target/Issuer)"] == "Brazil")
    if deal_min is not None:
        m &= df["valor"] > deal_min
    if pct_min is not None:
        m &= df["pct"] > pct_min
    out = df[m].copy()
    if dedup:
        out = out.drop_duplicates(
            ["Company Name (Target/Issuer)", "Announcement Date", "valor"])
    return out


def contagem_mensal(negocios: pd.DataFrame) -> pd.Series:
    """Nº de anúncios por mês (meses sem operação = 0)."""
    per = negocios["Announcement Date"].dt.to_period("M")
    s = per.value_counts().sort_index()
    idx = pd.period_range(INICIO, FIM, freq="M")
    return s.reindex(idx, fill_value=0).rename("N_MA")


def _serie_macro(m: pd.DataFrame, col_data: int, col_val: int, nome: str) -> pd.Series:
    s = m.iloc[:, [col_data, col_val]].copy()
    s.columns = ["data", nome]
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    s = s.dropna()
    s.index = s["data"].dt.to_period("M")
    return pd.to_numeric(s[nome], errors="coerce")


def carregar_macro() -> pd.DataFrame:
    """Séries mensais da aba Macro_Controls (posições das colunas na planilha)."""
    m = pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)
    macro = pd.concat([
        _serie_macro(m, 0, 1, "Selic"),       # Selic BCB, % a.m. acumulada no mês
        _serie_macro(m, 7, 8, "IPCA"),
        _serie_macro(m, 13, 14, "IBC_Br"),
        _serie_macro(m, 19, 20, "IIE_Br"),
        _serie_macro(m, 28, 29, "Ibov_Nivel"),
        _serie_macro(m, 28, 30, "Ibov_Vol"),
    ], axis=1)
    macro["Ibov_Ret"] = 100 * np.log(macro["Ibov_Nivel"]).diff()  # retorno mensal (%)
    return macro


def montar_painel(negocios: pd.DataFrame) -> pd.DataFrame:
    painel = pd.concat([contagem_mensal(negocios), carregar_macro()], axis=1)
    painel = painel.loc[INICIO:FIM]
    faltando = painel[VARIAVEIS].isna().sum()
    assert faltando.sum() == 0, f"Há valores ausentes no painel:\n{faltando}"
    painel.index.name = "Periodo"
    return painel


# =============================================================================
# 2. TESTES DE RAIZ UNITÁRIA
# =============================================================================


def testes_raiz_unitaria(painel: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for v in VARIAVEIS:
        for forma, serie in [("nível", painel[v]), ("1ª diferença", painel[v].diff().dropna())]:
            adf = adfuller(serie, regression="c", autolag="AIC")
            pp = PhillipsPerron(serie, trend="c")
            linhas.append({
                "Variável": v, "Forma": forma,
                "ADF stat": adf[0], "ADF p-valor": adf[1],
                "PP stat": pp.stat, "PP p-valor": pp.pvalue,
                "Estacionária (ADF e PP, 5%)": (adf[1] < .05) and (pp.pvalue < .05),
            })
    return pd.DataFrame(linhas)


def variaveis_nao_estacionarias(tabela: pd.DataFrame) -> list:
    niv = tabela[tabela["Forma"] == "nível"]
    return niv.loc[~niv["Estacionária (ADF e PP, 5%)"], "Variável"].tolist()


# =============================================================================
# 3. VAR: estimação, diagnóstico, Granger e IRF (Cholesky + bootstrap)
# =============================================================================


def selecionar_lags(dados: pd.DataFrame) -> pd.DataFrame:
    sel = VAR(dados).select_order(maxlags=MAXLAGS)
    return pd.DataFrame(sel.ics).rename_axis("lag")


def estimar_var(dados: pd.DataFrame, lags: int):
    return VAR(dados).fit(lags, trend="c")


def diagnosticos(res, lags: int) -> pd.DataFrame:
    branco = res.test_whiteness(nlags=max(lags + 8, 12), adjusted=True)
    normal = res.test_normality()
    return pd.DataFrame({
        "Teste": ["Estabilidade (raízes < 1)", "Ausência de autocorrelação (Portmanteau)",
                  "Normalidade dos resíduos (Jarque-Bera)"],
        "p-valor / resultado": ["estável" if res.is_stable() else "INSTÁVEL",
                                round(branco.pvalue, 4), round(normal.pvalue, 4)],
    })


def granger(res, dados: pd.DataFrame) -> pd.DataFrame:
    """Granger em bloco: cada variável -> N_MA e N_MA -> cada variável (teste F)."""
    linhas = []
    for v in [c for c in dados.columns if c != "N_MA"]:
        a = res.test_causality("N_MA", [v], kind="f")
        b = res.test_causality(v, ["N_MA"], kind="f")
        linhas.append({"Variável": v,
                       "Variável → N_MA (p)": a.pvalue,
                       "N_MA → Variável (p)": b.pvalue})
    return pd.DataFrame(linhas)


def _irf_ortogonal(coefs: np.ndarray, sigma: np.ndarray, passos: int) -> np.ndarray:
    """IRF de Cholesky: array (passos+1, resposta, choque)."""
    p, k, _ = coefs.shape
    phi = np.zeros((passos + 1, k, k))
    phi[0] = np.eye(k)
    for h in range(1, passos + 1):
        for j in range(1, min(h, p) + 1):
            phi[h] += phi[h - j] @ coefs[j - 1]
    P = np.linalg.cholesky(sigma)
    return np.einsum("hij,jk->hik", phi, P), P


def irf_bootstrap(res, dados: pd.DataFrame, choque_pp: float = CHOQUE_SELIC_PP):
    """
    IRF acumulada de Cholesky, com bootstrap dos resíduos (IC percentil).
    O choque em cada variável é escalado para 1 unidade da própria variável
    (ex.: +1 p.p. na Selic), como no paper-modelo.
    Retorna (irf, inferior, superior), arrays (H+1, resposta, choque).
    """
    rng = np.random.default_rng(SEMENTE)
    p = res.k_ar
    Y = dados.values
    T, k = Y.shape
    resid = res.resid.values
    const = res.params.values[0]                       # intercepto
    A = res.coefs

    def ponto(coefs, sigma):
        irf, P = _irf_ortogonal(coefs, sigma, HORIZONTE)
        escala = 1.0 / np.diag(P)                       # choque de 1 unidade
        return np.cumsum(irf * escala[None, None, :], axis=0)

    base = ponto(A, res.sigma_u.values)
    draws = np.empty((N_BOOT, HORIZONTE + 1, k, k))
    for b in range(N_BOOT):
        u = resid[rng.integers(0, len(resid), size=T - p)]
        Yb = np.zeros_like(Y)
        Yb[:p] = Y[:p]
        for t in range(p, T):
            Yb[t] = const + u[t - p] + sum(A[j] @ Yb[t - 1 - j] for j in range(p))
        rb = VAR(pd.DataFrame(Yb, columns=dados.columns)).fit(p, trend="c")
        draws[b] = ponto(rb.coefs, rb.sigma_u.values)
    a = (100 - NIVEL_IC) / 2
    return base, np.percentile(draws, a, axis=0), np.percentile(draws, 100 - a, axis=0)


def plotar_irf(base, lo, hi, dados, resposta="N_MA", arquivo="irf_N_MA.png"):
    i = list(dados.columns).index(resposta)
    choques = [c for c in dados.columns if c != resposta]
    fig, eixos = plt.subplots(1, len(choques), figsize=(4 * len(choques), 3.4), sharey=False)
    h = np.arange(HORIZONTE + 1)
    for ax, c in zip(np.atleast_1d(eixos), choques):
        j = list(dados.columns).index(c)
        ax.plot(h, base[:, i, j], color="#1f4e79", lw=2)
        ax.fill_between(h, lo[:, i, j], hi[:, i, j], color="#1f4e79", alpha=.18)
        ax.axhline(0, color="k", lw=.7)
        ax.set_title(f"Choque em {c}", fontsize=10)
        ax.set_xlabel("meses")
    np.atleast_1d(eixos)[0].set_ylabel(f"Resposta acumulada de {resposta}")
    fig.suptitle(f"IRF acumulada (Cholesky, IC {NIVEL_IC}% bootstrap)", y=1.02)
    fig.tight_layout()
    fig.savefig(SAIDA / arquivo, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resumo_irf(base, lo, hi, dados, choque, resposta="N_MA") -> dict:
    i, j = list(dados.columns).index(resposta), list(dados.columns).index(choque)
    return {"efeito_acum_h8": base[HORIZONTE, i, j],
            "IC_inf": lo[HORIZONTE, i, j], "IC_sup": hi[HORIZONTE, i, j],
            "significativo": not (lo[HORIZONTE, i, j] <= 0 <= hi[HORIZONTE, i, j])}


# =============================================================================
# 4. H3: ASSIMETRIA (VAR com Selic decomposta em alta e baixa acumuladas)
# =============================================================================


def dados_assimetricos(painel: pd.DataFrame) -> pd.DataFrame:
    """Substitui a Selic por Selic⁺ e Selic⁻ (somas acumuladas das variações
    positivas e negativas), como em Hatemi-J (2012). Permite testar se uma
    alta e uma queda de juros afetam o M&A de forma diferente."""
    d = painel[[v for v in VARIAVEIS if v != "Selic"]].copy()
    ds = painel["Selic"].diff().fillna(0)
    d["Selic_Alta"] = ds.clip(lower=0).cumsum()
    d["Selic_Queda"] = ds.clip(upper=0).cumsum()
    return d


def rodar_h3(painel: pd.DataFrame, lags: int) -> pd.DataFrame:
    d = dados_assimetricos(painel)
    res = estimar_var(d, lags)
    base, lo, hi = irf_bootstrap(res, d)
    plotar_irf(base, lo, hi, d, arquivo="irf_H3_assimetria.png")
    linhas = []
    for c in ["Selic_Alta", "Selic_Queda"]:
        linhas.append({"Choque": c, **resumo_irf(base, lo, hi, d, c)})
    # Teste de igualdade de coeficientes na equação de N_MA (OLS auxiliar)
    X = pd.concat([d.shift(k).add_suffix(f"_L{k}") for k in range(1, lags + 1)], axis=1).dropna()
    y = d["N_MA"].loc[X.index]
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    restr = ", ".join(f"Selic_Alta_L{k} = Selic_Queda_L{k}" for k in range(1, lags + 1))
    wt = ols.wald_test(restr, scalar=True)
    linhas.append({"Choque": "Wald: Alta = Queda (eq. N_MA)",
                   "efeito_acum_h8": np.nan, "IC_inf": np.nan, "IC_sup": np.nan,
                   "significativo": bool(wt.pvalue < .05), "p-valor Wald": float(wt.pvalue)})
    return pd.DataFrame(linhas)


# =============================================================================
# 5. ROBUSTEZ
# =============================================================================


def preparar(painel, transformacao, cols=None, ordem=None):
    cols = cols or VARIAVEIS
    d = painel[cols].copy()
    if ordem:
        d = d[ordem]
    if transformacao == "diferencas":
        d = d.diff().dropna()
    return d


def especificacao(nome, painel_, transformacao, lags, cols=None, ordem=None):
    d = preparar(painel_, transformacao, cols, ordem)
    res = estimar_var(d, lags)
    base, lo, hi = irf_bootstrap(res, d)
    linha = {"Especificação": nome, "lags": lags, "obs": res.nobs}
    if "Selic" in d.columns:
        r = resumo_irf(base, lo, hi, d, "Selic")
        linha.update({"Selic→N_MA (h=8)": r["efeito_acum_h8"],
                      "Selic IC": f"[{r['IC_inf']:.2f}; {r['IC_sup']:.2f}]"})
    if "IIE_Br" in d.columns:
        r = resumo_irf(base, lo, hi, d, "IIE_Br")
        linha.update({"IIE→N_MA (h=8)": r["efeito_acum_h8"],
                      "IIE IC": f"[{r['IC_inf']:.2f}; {r['IC_sup']:.2f}]"})
    return linha


def robustez(painel, base_df_negocios, lags):
    linhas = [
        especificacao("Principal", painel, TRANSFORMACAO, lags),
        especificacao("Selic ordenada primeiro", painel, TRANSFORMACAO, lags,
                      ordem=["Selic"] + [v for v in VARIAVEIS if v != "Selic"]),
        especificacao("Sem IIE-Br (5 variáveis)", painel, TRANSFORMACAO, lags,
                      cols=[v for v in VARIAVEIS if v != "IIE_Br"]),
        especificacao("Variáveis em 1ª diferença", painel, "diferencas", lags),
        especificacao(f"VAR({lags + 1})", painel, TRANSFORMACAO, lags + 1),
    ]
    # Critérios alternativos de amostra
    alternativas = {
        "Amostra: sem filtro de %": dict(pct_min=None),
        "Amostra: sem filtro de valor": dict(deal_min=None),
        "Amostra: valor > US$ 5 MM": dict(deal_min=5.0),
        "Amostra: sem duplicatas": dict(dedup=True),
    }
    for nome, kw in alternativas.items():
        neg = aplicar_filtros(base_df_negocios, **kw)
        p = painel.copy()
        p["N_MA"] = contagem_mensal(neg)
        linhas.append(especificacao(nome, p, TRANSFORMACAO, lags))
    # Subamostra sem crise / pandemia
    sub = painel.loc[:"2019-12"]
    linhas.append(especificacao("Subamostra 2005-2019 (sem pandemia)", sub, TRANSFORMACAO, lags))
    return pd.DataFrame(linhas)


# =============================================================================
# 6. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 70)
    print("1. DADOS")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final: {len(negocios):,}")
    dup = negocios.duplicated(["Company Name (Target/Issuer)", "Announcement Date", "valor"],
                              keep=False).sum()
    print(f"Linhas potencialmente duplicadas na amostra: {dup}")

    painel = montar_painel(negocios)
    painel.to_csv(SAIDA / "painel_mensal.csv")
    desc = painel[VARIAVEIS].describe().T
    desc.to_csv(SAIDA / "estatisticas_descritivas.csv")
    print(f"Painel: {painel.index[0]} a {painel.index[-1]} ({len(painel)} meses)")
    print(desc.round(2))

    print("\n2. RAIZ UNITÁRIA (ADF e Phillips-Perron)")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria.csv", index=False)
    print(ru.round(3).to_string(index=False))
    nao_est = variaveis_nao_estacionarias(ru)
    print(f"\nNão estacionárias em nível: {nao_est or 'nenhuma'}")
    if nao_est and TRANSFORMACAO == "niveis":
        print("AVISO: há variáveis I(1). O modelo principal segue em níveis (réplica do "
              "paper); a robustez em 1ª diferença é reportada na secção 5.")

    dados = preparar(painel, TRANSFORMACAO)

    print("\n3. SELEÇÃO DE DEFASAGENS")
    ics = selecionar_lags(dados)
    ics.to_csv(SAIDA / "selecao_lags.csv")
    print(ics.round(3))
    lags = LAGS_PRINCIPAL or int(ics["bic"].idxmin())
    lags = max(lags, 1)
    print(f"Defasagens usadas: {lags}  (AIC sugere {int(ics['aic'].idxmin())})")

    print("\n4. ESTIMAÇÃO DO VAR")
    res = estimar_var(dados, lags)
    with open(SAIDA / "var_resumo.txt", "w", encoding="utf-8") as f:
        f.write(str(res.summary()))
    print(diagnosticos(res, lags).to_string(index=False))

    print("\n5. CAUSALIDADE DE GRANGER")
    g = granger(res, dados)
    g.to_csv(SAIDA / "granger.csv", index=False)
    print(g.round(4).to_string(index=False))

    print(f"\n6. IRF (Cholesky, bootstrap {N_BOOT}x, choque de 1 unidade)")
    base, lo, hi = irf_bootstrap(res, dados)
    plotar_irf(base, lo, hi, dados)
    h12 = pd.DataFrame([
        {"Choque": c, **resumo_irf(base, lo, hi, dados, c)}
        for c in dados.columns if c != "N_MA"
    ])
    h12.to_csv(SAIDA / "irf_resumo_N_MA.csv", index=False)
    print(h12.round(3).to_string(index=False))
    print("\n-> H1: veja a linha 'Selic' (efeito acumulado negativo e IC sem zero = suporte)")
    print("-> H2: veja a linha 'IIE_Br' (efeito significativo mesmo controlando pela Selic)")

    print("\n7. H3 - ASSIMETRIA (Selic⁺ vs Selic⁻)")
    h3 = rodar_h3(painel, lags)
    h3.to_csv(SAIDA / "h3_assimetria.csv", index=False)
    print(h3.round(3).to_string(index=False))

    print("\n8. ROBUSTEZ")
    rb = robustez(painel, bruto, lags)
    rb.to_csv(SAIDA / "robustez.csv", index=False)
    print(rb.round(2).to_string(index=False))

    print(f"\nPronto. Resultados em: {SAIDA}")


if __name__ == "__main__":
    main()
