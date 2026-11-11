"""
A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
Evidências dos Ciclos da Selic (2005-2024)

Versão VAR — Selic-surpresa — cenário adicional ao VAR original
(Código_TCC_VAR.py):
  - Troca ΔSelic (variação bruta) pela SURPRESA da decisão do Copom: a
    diferença entre a Selic efetivamente decidida em cada reunião e a
    mediana das expectativas de mercado (Boletim Focus/BCB) coletada até a
    véspera da reunião.
  - Mesma estrutura de VAR da versão anterior (4 endógenas + Covid exógena,
    seleção de defasagem, Granger, IRF, FEVD, robustez de ordenação).

Por quê: a versão anterior mostrou causalidade de Granger BIDIRECIONAL entre
Selic e M&A, e uma resposta contemporânea (mesmo mês) sensível à ordenação de
Cholesky — os dois sinais de que ΔSelic bruta está contaminada por
simultaneidade (o Copom reage às mesmas condições que movem o M&A). A
surpresa do Copom, por só capturar a parte da decisão que o mercado NÃO
esperava, é uma aproximação mais defensável de um choque exógeno de política
monetária — é o mesmo princípio usado na literatura internacional de "policy
surprises" (ex. Kuttner 2001, Gürkaynak-Sack-Swanson 2005) e endereça
diretamente a limitação identificada na análise de robustez anterior.

FONTE DE DADOS NOVA — Boletim Focus (BCB), API pública "Expectativas de
Mercado": https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoSelic
Busca, para cada reunião do Copom, a mediana das expectativas de Selic
coletadas na véspera; compara com a Selic efetivamente decidida (série SGS
432 do BCB, https://api.bcb.gov.br/dados/serie/bcdata.sgs.432).

*** AVISO IMPORTANTE ***
O ambiente onde este código foi escrito e testado (sandbox de nuvem do
Claude) bloqueia o domínio olinda.bcb.gov.br por política de rede — não foi
possível baixar os dados do Focus nem rodar este script até o fim aqui. A
lógica de download, o schema da API (campos Indicador/Data/Reuniao/Mediana/
baseCalculo) e o merge foram verificados via documentação oficial e da
biblioteca `python-bcb`, mas os RESULTADOS NUMÉRICOS deste cenário ainda não
foram gerados nem conferidos. Rode este script no seu computador (seu
ambiente não tem essa restrição) e me envie a saída (ou o arquivo
`focus_selic_mediana.csv` gerado) para eu revisar e escrever a análise.

Se preferir não depender da API em toda execução, rode uma vez com
BAIXAR_FOCUS=True para gerar o cache local (`focus_selic_mediana.csv`
dentro de ./dados_focus/) e depois pode rodar com BAIXAR_FOCUS=False.

Saídas em ./resultados_var_surpresa.
"""

from pathlib import Path
import sys
import time
import warnings
from urllib.parse import urlencode, quote

# O console do Windows costuma usar um codepage (ex.: cp1252) que não tem
# caracteres como "Δ" — sem isso, um print com esse tipo de caractere
# derruba o script com UnicodeEncodeError mesmo depois de tudo já ter sido
# salvo em disco.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados_var_surpresa"
SAIDA.mkdir(exist_ok=True)
DIR_FOCUS = BASE / "dados_focus"
DIR_FOCUS.mkdir(exist_ok=True)
CACHE_FOCUS = DIR_FOCUS / "focus_selic_mediana.csv"
CACHE_SELIC_SGS = DIR_FOCUS / "selic_meta_sgs432.csv"

BAIXAR_FOCUS = True   # False usa o cache local (CACHE_FOCUS/CACHE_SELIC_SGS), se existir

DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0

INICIO, FIM = "2005-01", "2024-12"
COVID_INICIO, COVID_FIM = "2020-03", "2020-12"

MAXLAGS = 12
PERIODOS_IRF = 12
BASE_CALCULO_FOCUS = 0   # 0 = base de cálculo padrão do BCB (ver documentação da API)

URL_FOCUS = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
            "ExpectativasMercadoSelic")
URL_SGS_432 = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"

# =============================================================================
# 1. SELIC-SURPRESA — download e construção
# =============================================================================


def baixar_focus_selic_mediana() -> pd.DataFrame:
    """
    Baixa, para cada reunião do Copom, a série completa de medianas de
    expectativa de Selic coletadas dia a dia (endpoint ExpectativasMercadoSelic,
    paginado em blocos de 1000 via $skip/$top — a API OData do BCB limita o
    retorno por página). Colunas retornadas pela API: Indicador, Data, Reuniao,
    Media, Mediana, DesvioPadrao, Minimo, Maximo, numeroRespondentes,
    baseCalculo.
    """
    linhas = []
    skip = 0
    passo = 1000
    while True:
        params = {
            "$format": "json",
            "$top": passo,
            "$skip": skip,
            "$filter": f"baseCalculo eq {BASE_CALCULO_FOCUS}",
            "$select": "Data,Reuniao,Mediana,Media,numeroRespondentes",
            "$orderby": "Data asc",
        }
        # A API OData do BCB não decodifica corretamente espaços codificados
        # como "+" (o padrão do requests/urlencode) dentro do $filter — exige
        # "%20" literal, senão devolve um 400 "Edm.Boolean incompatível" sem
        # relação aparente com o filtro em si. Por isso montamos a query
        # manualmente com quote_via=quote.
        query = urlencode(params, quote_via=quote)
        resp = requests.get(f"{URL_FOCUS}?{query}", timeout=30)
        resp.raise_for_status()
        bloco = resp.json().get("value", [])
        if not bloco:
            break
        linhas.extend(bloco)
        skip += passo
        time.sleep(0.2)  # gentileza com a API pública
        if len(bloco) < passo:
            break
    df = pd.DataFrame(linhas)
    df["Data"] = pd.to_datetime(df["Data"])
    df.to_csv(CACHE_FOCUS, index=False)
    return df


def baixar_selic_decidida() -> pd.DataFrame:
    """Série SGS 432 — Meta Selic definida pelo Copom, diária (constante entre
    reuniões, muda no dia seguinte a cada decisão).

    A API do SGS recusa (406) qualquer consulta de série diária que cubra
    mais de 10 anos sem dataInicial/dataFinal explícitos — e mesmo com elas,
    uma janela maior que 10 anos ainda é recusada. Como o período do estudo
    (2005-2024) tem ~20 anos, a série é baixada em blocos de até 10 anos e
    concatenada. Também envia um User-Agent explícito e faz uma nova
    tentativa em caso de erro transitório do servidor (observado como uma
    página HTML "Requisição inválida" ocasional, não relacionada aos
    parâmetros da consulta).
    """
    inicio = pd.Period(INICIO).start_time - pd.Timedelta(days=10)
    fim = pd.Period(FIM).end_time + pd.Timedelta(days=10)
    headers = {"User-Agent": "Mozilla/5.0"}

    blocos = []
    cursor = inicio
    while cursor <= fim:
        fim_bloco = min(cursor + pd.DateOffset(years=10) - pd.Timedelta(days=1), fim)
        params = {
            "formato": "json",
            "dataInicial": cursor.strftime("%d/%m/%Y"),
            "dataFinal": fim_bloco.strftime("%d/%m/%Y"),
        }
        for tentativa in range(6):
            resp = requests.get(URL_SGS_432, params=params, headers=headers, timeout=60)
            if resp.ok and resp.text.strip().startswith(("[", "{")):
                break
            time.sleep(3 * (tentativa + 1))  # backoff: 3s, 6s, 9s, ... — o servidor do
            # BCB às vezes devolve uma página HTML de erro ("Requisição inválida") de
            # forma transitória/intermitente, sem relação com os parâmetros enviados.
        else:
            raise RuntimeError(
                f"SGS 432 não respondeu com JSON válido após 6 tentativas para "
                f"{params['dataInicial']}–{params['dataFinal']}. Último corpo: "
                f"{resp.text[:200]!r}")
        resp.raise_for_status()
        blocos.append(pd.DataFrame(resp.json()))
        cursor = fim_bloco + pd.Timedelta(days=1)
        time.sleep(0.2)

    df = pd.concat(blocos, ignore_index=True).drop_duplicates(subset="data")
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"])
    df = df.rename(columns={"data": "Data", "valor": "Selic_decidida"}).sort_values("Data")
    df.to_csv(CACHE_SELIC_SGS, index=False)
    return df


def construir_selic_surpresa() -> pd.DataFrame:
    """
    Para cada reunião do Copom (campo `Reuniao`):
      1. pega a última `Mediana` disponível antes da reunião (expectativa de
         mercado na véspera da decisão);
      2. pega a Selic decidida (série 432) no primeiro dia útil após essa
         data, dentro de uma janela de 5 dias;
      3. surpresa = Selic decidida - expectativa mediana.
    Retorna um DataFrame com uma linha por reunião: Data (da expectativa),
    Reuniao, Selic_esperada, Selic_decidida, Surpresa.
    """
    if BAIXAR_FOCUS or not CACHE_FOCUS.exists():
        focus = baixar_focus_selic_mediana()
    else:
        focus = pd.read_csv(CACHE_FOCUS, parse_dates=["Data"])

    if BAIXAR_FOCUS or not CACHE_SELIC_SGS.exists():
        selic_sgs = baixar_selic_decidida()
    else:
        selic_sgs = pd.read_csv(CACHE_SELIC_SGS, parse_dates=["Data"])

    ultima_por_reuniao = (
        focus.sort_values("Data")
        .groupby("Reuniao", as_index=False)
        .last()[["Reuniao", "Data", "Mediana"]]
        .rename(columns={"Mediana": "Selic_esperada"})
    )

    selic_sgs = selic_sgs.sort_values("Data").reset_index(drop=True)

    def _selic_decidida_apos(data_ref, janela_dias=5):
        janela = selic_sgs[(selic_sgs["Data"] > data_ref) &
                           (selic_sgs["Data"] <= data_ref + pd.Timedelta(days=janela_dias))]
        return janela["Selic_decidida"].iloc[0] if len(janela) else np.nan

    ultima_por_reuniao["Selic_decidida"] = ultima_por_reuniao["Data"].apply(_selic_decidida_apos)
    ultima_por_reuniao["Surpresa"] = (
        ultima_por_reuniao["Selic_decidida"] - ultima_por_reuniao["Selic_esperada"])
    ultima_por_reuniao = ultima_por_reuniao.dropna(subset=["Surpresa"])
    ultima_por_reuniao.to_csv(SAIDA / "selic_surpresa_por_reuniao.csv", index=False)
    return ultima_por_reuniao


def agregar_surpresa_mensal(surpresas: pd.DataFrame) -> pd.Series:
    """Agrega a surpresa (evento, ocorre em ~8 datas/ano) para o mês em que a
    decisão foi tomada; meses sem reunião do Copom recebem 0 (sem choque de
    política monetária além do esperado nesse mês)."""
    s = surpresas.copy()
    s["Periodo"] = s["Data"].dt.to_period("M")
    mensal = s.groupby("Periodo")["Surpresa"].sum()
    idx = pd.period_range(INICIO, FIM, freq="M")
    return mensal.reindex(idx, fill_value=0.0).rename("Selic_surpresa")


# =============================================================================
# 2. DADOS DO M&A — mesmo pipeline das versões 1-4 e do VAR original
# =============================================================================


def carregar_negocios() -> pd.DataFrame:
    df = pd.read_excel(ARQUIVO, sheet_name="Transactions_Results", header=2)
    df["Announcement Date"] = pd.to_datetime(df["Announcement Date"])
    df["valor"] = pd.to_numeric(df["Deal Size (USD, MM)"], errors="coerce")
    df["pct"] = pd.to_numeric(df["% of Target Shares Sought"], errors="coerce")
    return df


def aplicar_filtros(df, deal_min=DEAL_MIN_USD_MM, pct_min=PCT_MIN):
    m = (df["Deal Status"] == "Completed") & (df["Country (Target/Issuer)"] == "Brazil")
    m &= df["valor"] > deal_min
    m &= df["pct"] > pct_min
    return df[m].copy()


def contagem_mensal(negocios: pd.DataFrame) -> pd.Series:
    per = negocios["Announcement Date"].dt.to_period("M")
    s = per.value_counts().sort_index()
    idx = pd.period_range(INICIO, FIM, freq="M")
    return s.reindex(idx, fill_value=0).rename("N_MA")


def _serie_macro(m, col_data, col_val, nome):
    s = m.iloc[:, [col_data, col_val]].copy()
    s.columns = ["data", nome]
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    s = s.dropna()
    s.index = s["data"].dt.to_period("M")
    return pd.to_numeric(s[nome], errors="coerce")


def carregar_macro() -> pd.DataFrame:
    m = pd.read_excel(ARQUIVO, sheet_name="Macro_Controls", header=2)
    macro = pd.concat([
        _serie_macro(m, 13, 14, "IBC_Br"),
        _serie_macro(m, 28, 29, "Ibov_Nivel"),
    ], axis=1)
    macro["Ibov_Ret"] = 100 * np.log(macro["Ibov_Nivel"]).diff()
    return macro


def montar_painel(negocios: pd.DataFrame, surpresa_mensal: pd.Series) -> pd.DataFrame:
    painel = pd.concat([contagem_mensal(negocios), carregar_macro(), surpresa_mensal], axis=1)
    painel = painel.loc[INICIO:FIM]
    painel["log_N_MA"] = np.log(painel["N_MA"])
    painel["dIBC_Br"] = painel["IBC_Br"].pct_change() * 100
    idx_str = painel.index.astype(str)
    painel["Covid"] = ((idx_str >= COVID_INICIO) & (idx_str <= COVID_FIM)).astype(int)
    painel.index.name = "Periodo"
    return painel


ENDOGENAS = ["log_N_MA", "Selic_surpresa", "dIBC_Br", "Ibov_Ret"]


def testes_raiz_unitaria(painel: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for nome in ENDOGENAS:
        serie = painel[nome].dropna()
        adf = adfuller(serie, regression="c", autolag="AIC")
        linhas.append({"Variável": nome, "ADF stat": adf[0], "ADF p-valor": adf[1],
                       "Estacionária a 5%": adf[1] < .05})
    return pd.DataFrame(linhas)


def preparar_dados_var(painel: pd.DataFrame):
    dados = painel[ENDOGENAS + ["Covid"]].dropna().copy()
    y = dados[ENDOGENAS]
    # NÃO usar sm.add_constant aqui: o VAR do statsmodels já inclui uma
    # constante por padrão (trend="c"). Somar uma segunda coluna de
    # constante via exog cria duas colunas idênticas de 1s no design
    # matrix (endog_lagged) -> matriz exatamente deficiente em posto ->
    # cov_params()/summary() falha com "Singular matrix" (ou, em amostras
    # menores/p menor, roda sem erro mas devolve std. error/t-stat/p-valor
    # = NaN para a constante, como se viu no VAR original em
    # resultados_var/var_p2_sumario.txt, que tem "const" duplicado com
    # NAN). Passar só a(s) variável(is) exógena(s) resolve.
    exog = dados[["Covid"]]
    return y, exog, dados


def selecionar_defasagem(y: pd.DataFrame, exog: pd.DataFrame):
    modelo = VAR(y, exog=exog)
    sel = modelo.select_order(maxlags=MAXLAGS)
    tab = pd.DataFrame(sel.summary().data[1:], columns=sel.summary().data[0])
    return tab, sel


def estabilidade(resultado) -> pd.DataFrame:
    mods = np.abs(resultado.roots)
    return pd.DataFrame({"raiz (módulo)": mods, "estável (>1)": mods > 1})


def causalidade_granger(dados: pd.DataFrame, p: int) -> pd.DataFrame:
    """Implementação manual (OLS aninhado), igual à do VAR original — evita o
    bug do test_causality nativo do statsmodels 0.15.0 identificado antes."""
    linhas = []
    for causing in ENDOGENAS:
        for caused in ENDOGENAS:
            if causing == caused:
                continue
            df = dados.copy()
            cols_irrestrito = []
            for var in ENDOGENAS:
                for lag in range(1, p + 1):
                    col = f"{var}_L{lag}"
                    df[col] = df[var].shift(lag)
                    cols_irrestrito.append(col)
            df = df.dropna()
            y = df[caused]
            X_irrestrito = sm.add_constant(df[["Covid"] + cols_irrestrito])
            cols_restrito = [c for c in cols_irrestrito if not c.startswith(f"{causing}_L")]
            X_restrito = sm.add_constant(df[["Covid"] + cols_restrito])
            m_irrestrito = sm.OLS(y, X_irrestrito).fit()
            m_restrito = sm.OLS(y, X_restrito).fit()
            F, pval, _ = m_irrestrito.compare_f_test(m_restrito)
            linhas.append({"causa (Granger)": causing, "variável afetada": caused,
                           "estatística F": F, "p-valor": pval,
                           "significativo a 5%": bool(pval < .05)})
    return pd.DataFrame(linhas)


def whiteness_residuos(resultado, nlags=12) -> dict:
    nlags = max(nlags, resultado.k_ar + 1)
    teste = resultado.test_whiteness(nlags=nlags)
    return {"estatística (Portmanteau)": teste.test_statistic, "p-valor": teste.pvalue,
           "resíduos sem autocorrelação a 5%": teste.pvalue > .05}


def extrair_irf_surpresa_para_ma(resultado, periodos=PERIODOS_IRF, alpha=0.05) -> pd.DataFrame:
    irf = resultado.irf(periods=periodos)
    i_surp = ENDOGENAS.index("Selic_surpresa")
    i_ma = ENDOGENAS.index("log_N_MA")
    resp = irf.orth_irfs[:, i_ma, i_surp]
    ic_baixo, ic_alto = irf.errband_mc(orth=True, repl=1000, signif=alpha)
    linhas = []
    for h in range(len(resp)):
        linhas.append({"horizonte (meses)": h,
                       "resposta log(N_MA) a choque +1dp em Selic-surpresa": resp[h],
                       "IC inferior": ic_baixo[h, i_ma, i_surp],
                       "IC superior": ic_alto[h, i_ma, i_surp]})
    return pd.DataFrame(linhas), irf


def grafico_irf(irf, caminho):
    fig = irf.plot(orth=True, signif=0.05, impulse="Selic_surpresa", response="log_N_MA")
    fig.set_size_inches(9, 5)
    fig.suptitle("Resposta de log(N_MA) a um choque ortogonalizado em Selic-surpresa (IC 95%)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def decomposicao_variancia(resultado, periodos=PERIODOS_IRF) -> pd.DataFrame:
    fevd = resultado.fevd(periods=periodos)
    i_ma = ENDOGENAS.index("log_N_MA")
    tab = pd.DataFrame(fevd.decomp[i_ma], columns=ENDOGENAS)
    tab.index.name = "horizonte (meses)"
    return tab


ORDENACOES_TESTADAS = {
    "Original (M&A mais exógeno)": ["log_N_MA", "Selic_surpresa", "dIBC_Br", "Ibov_Ret"],
    "Surpresa mais exógena": ["Selic_surpresa", "log_N_MA", "dIBC_Br", "Ibov_Ret"],
    "Atividade econômica primeiro": ["dIBC_Br", "Selic_surpresa", "log_N_MA", "Ibov_Ret"],
    "Mercado financeiro primeiro": ["Ibov_Ret", "Selic_surpresa", "dIBC_Br", "log_N_MA"],
}


def robustez_ordenacao(y, exog, p, periodos=PERIODOS_IRF) -> pd.DataFrame:
    linhas = []
    for nome_ordem, ordem in ORDENACOES_TESTADAS.items():
        modelo = VAR(y[ordem], exog=exog)
        resultado = modelo.fit(p)
        irf = resultado.irf(periods=periodos)
        i_s, i_ma = ordem.index("Selic_surpresa"), ordem.index("log_N_MA")
        resp = irf.orth_irfs[:, i_ma, i_s]
        for h, valor in enumerate(resp):
            linhas.append({"ordenação": nome_ordem, "horizonte (meses)": h,
                           "resposta log(N_MA) a choque +1dp em Selic-surpresa": valor})
    return pd.DataFrame(linhas)


def grafico_series(painel: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    dados = {"log_N_MA": "log(N_MA)", "Selic_surpresa": "Selic-surpresa (p.p.)",
            "dIBC_Br": "ΔIBC-Br (%)", "Ibov_Ret": "Retorno Ibovespa (%)"}
    cores = ["#1f4e79", "#b7472a", "#6a3d9a", "#2e7d32"]
    for ax, (col, titulo), cor in zip(axes, dados.items(), cores):
        ax.plot(painel.index.to_timestamp(), painel[col], color=cor)
        ax.set_title(titulo)
        if col != "log_N_MA":
            ax.axhline(0, color="k", lw=.5)
        ax.axvspan(pd.Period(COVID_INICIO).to_timestamp(), pd.Period(COVID_FIM).to_timestamp(),
                  color="gray", alpha=.15)
    fig.tight_layout()
    fig.savefig(SAIDA / "series_var_surpresa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 3. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 70)
    print("1. SELIC-SURPRESA (Focus/BCB)")
    surpresas = construir_selic_surpresa()
    print(f"{len(surpresas)} reuniões do Copom com surpresa calculada "
          f"({surpresas['Data'].min().date()} a {surpresas['Data'].max().date()})")
    surpresa_mensal = agregar_surpresa_mensal(surpresas)

    print("\n2. DADOS DE M&A")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final: {len(negocios):,}")

    painel = montar_painel(negocios, surpresa_mensal)
    painel.to_csv(SAIDA / "painel_mensal_surpresa.csv")
    grafico_series(painel)

    print("\n3. RAIZ UNITÁRIA")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria.csv", index=False)
    print(ru.round(3).to_string(index=False))

    y, exog, dados = preparar_dados_var(painel)
    print(f"\nAmostra do VAR: n={len(dados)} meses")

    print("\n4. SELEÇÃO DE DEFASAGEM")
    tab_sel, sel = selecionar_defasagem(y, exog)
    tab_sel.to_csv(SAIDA / "selecao_defasagem.csv", index=False)
    print(tab_sel.to_string(index=False))
    p_escolhido = sel.bic if sel.bic and sel.bic >= 1 else max(sel.aic, 1)
    TETO_DEFASAGEM = 8  # segurança: evita matriz quase singular com poucos dados por parâmetro
    if p_escolhido > TETO_DEFASAGEM:
        print(f"BIC sugeriu p={p_escolhido}, acima do teto de segurança ({TETO_DEFASAGEM}) para "
              f"esta amostra — usando p={TETO_DEFASAGEM} em vez disso.")
        p_escolhido = TETO_DEFASAGEM
    print(f"Defasagem adotada: p={p_escolhido}")

    print(f"\n5. ESTIMAÇÃO DO VAR({p_escolhido})")
    modelo = VAR(y, exog=exog)
    resultado = modelo.fit(p_escolhido)
    with open(SAIDA / f"var_surpresa_p{p_escolhido}_sumario.txt", "w", encoding="utf-8") as f:
        f.write(str(resultado.summary()))
    print(f"AIC={resultado.aic:.3f}  BIC={resultado.bic:.3f}")

    print("\n6. ESTABILIDADE")
    est = estabilidade(resultado)
    est.to_csv(SAIDA / "estabilidade.csv", index=False)
    print("Estável." if est["estável (>1)"].all() else "ATENÇÃO: possivelmente instável.")

    print("\n7. AUTOCORRELAÇÃO DOS RESÍDUOS")
    white = whiteness_residuos(resultado)
    pd.Series(white).to_csv(SAIDA / "whiteness_residuos.csv")
    print(white)

    print("\n8. CAUSALIDADE DE GRANGER")
    granger = causalidade_granger(dados, p_escolhido)
    granger.to_csv(SAIDA / "causalidade_granger.csv", index=False)
    print(granger.round(4).to_string(index=False))

    print("\n9. IMPULSO-RESPOSTA — Selic-surpresa -> log(N_MA)")
    tab_irf, irf = extrair_irf_surpresa_para_ma(resultado)
    tab_irf.to_csv(SAIDA / "irf_surpresa_para_ma.csv", index=False)
    print(tab_irf.round(5).to_string(index=False))
    grafico_irf(irf, SAIDA / "irf_surpresa_para_logma.png")

    print("\n10. DECOMPOSIÇÃO DA VARIÂNCIA")
    fevd = decomposicao_variancia(resultado)
    fevd.to_csv(SAIDA / "fevd_log_ma.csv")
    print(fevd.round(4).to_string())

    print("\n11. ROBUSTEZ — SENSIBILIDADE À ORDENAÇÃO DE CHOLESKY")
    tab_rob = robustez_ordenacao(y, exog, p_escolhido)
    tab_rob.to_csv(SAIDA / "robustez_ordenacao.csv", index=False)
    print("Salvo em robustez_ordenacao.csv (ver gráfico comparativo com o VAR de ΔSelic bruta).")

    print(f"\nPronto. Resultados em: {SAIDA}")


if __name__ == "__main__":
    main()
