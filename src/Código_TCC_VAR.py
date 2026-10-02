"""
A Política Monetária afeta o volume de Fusões e Aquisições no Brasil?
Evidências dos Ciclos da Selic (2005-2024)

Versão VAR — cenário alternativo ao MQO das versões 1-4:
  - Sistema de Vetores Autorregressivos (VAR) com log(M&A), ΔSelic, ΔIBC-Br
    e Ibov_Ret como endógenas, Covid como exógena.
  - Seleção do número de defasagens por AIC/BIC/HQIC/FPE.
  - Causalidade de Granger nos dois sentidos (Selic -> M&A e M&A -> Selic).
  - Funções de impulso-resposta (IRF) ortogonalizadas com IC 95% (bootstrap).
  - Decomposição da variância do erro de previsão (FEVD).
  - Checagem de estabilidade (raízes do processo) e de autocorrelação dos
    resíduos (teste de Portmanteau).
  - Robustez: sensibilidade da IRF a diferentes ordenações de Cholesky
    (a IRF ortogonalizada depende de qual variável é tratada como mais
    "exógena" contemporaneamente — aqui testam-se 4 ordenações plausíveis).

Inspirado em Souza, Gil e Triches (2024), que usa causalidade de Granger
entre Selic e M&A no Brasil (1994-2021, dados anuais) e sugere, nas
conclusões, compor um VAR com mais variáveis — é o que se faz aqui, em
painel mensal (2005-2024) e com mais uma variável (Ibovespa).

Não substitui as versões 1-4 (MQO) — é um cenário complementar, porque VAR
trata todas as variáveis como endógenas e permite checar causalidade nos
dois sentidos, em vez de assumir de partida que a Selic é exógena ao M&A.
Saídas em ./resultados_var.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / "Deals_Results_TCC_FactSet_Att.xlsx"
SAIDA = BASE / "resultados_var"
SAIDA.mkdir(exist_ok=True)

DEAL_MIN_USD_MM = 0.5
PCT_MIN = 50.0

INICIO, FIM = "2005-01", "2024-12"
COVID_INICIO, COVID_FIM = "2020-03", "2020-12"

MAXLAGS = 12          # teto para a seleção do número de defasagens
PERIODOS_IRF = 12     # horizonte da função de impulso-resposta (meses)

# =============================================================================
# 1. DADOS — mesmo pipeline das versões 1-4 (filtros, painel mensal)
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
        _serie_macro(m, 0, 1, "Selic"),
        _serie_macro(m, 7, 8, "IPCA"),
        _serie_macro(m, 13, 14, "IBC_Br"),
        _serie_macro(m, 19, 20, "IIE_Br"),
        _serie_macro(m, 28, 29, "Ibov_Nivel"),
        _serie_macro(m, 28, 30, "Ibov_Vol"),
    ], axis=1)
    macro["Ibov_Ret"] = 100 * np.log(macro["Ibov_Nivel"]).diff()
    return macro


def montar_painel(negocios: pd.DataFrame) -> pd.DataFrame:
    painel = pd.concat([contagem_mensal(negocios), carregar_macro()], axis=1)
    painel = painel.loc[INICIO:FIM]
    painel["log_N_MA"] = np.log(painel["N_MA"])
    painel["dSelic"] = painel["Selic"].diff()
    painel["dIBC_Br"] = painel["IBC_Br"].pct_change() * 100
    idx_str = painel.index.astype(str)
    painel["Covid"] = ((idx_str >= COVID_INICIO) & (idx_str <= COVID_FIM)).astype(int)
    painel.index.name = "Periodo"
    return painel


# =============================================================================
# 2. ESTACIONARIEDADE DAS SÉRIES DO SISTEMA
# =============================================================================

ENDOGENAS = ["log_N_MA", "dSelic", "dIBC_Br", "Ibov_Ret"]


def testes_raiz_unitaria(painel: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for nome in ENDOGENAS:
        serie = painel[nome].dropna()
        adf = adfuller(serie, regression="c", autolag="AIC")
        linhas.append({"Variável": nome, "ADF stat": adf[0], "ADF p-valor": adf[1],
                       "Estacionária a 5%": adf[1] < .05})
    return pd.DataFrame(linhas)


# =============================================================================
# 3. VAR — SELEÇÃO DE DEFASAGEM, ESTIMAÇÃO, ESTABILIDADE
# =============================================================================


def preparar_dados_var(painel: pd.DataFrame):
    dados = painel[ENDOGENAS + ["Covid"]].dropna().copy()
    y = dados[ENDOGENAS]
    exog = sm.add_constant(dados[["Covid"]])
    return y, exog, dados


def selecionar_defasagem(y: pd.DataFrame, exog: pd.DataFrame) -> pd.DataFrame:
    modelo = VAR(y, exog=exog)
    sel = modelo.select_order(maxlags=MAXLAGS)
    tab = pd.DataFrame(sel.summary().data[1:], columns=sel.summary().data[0])
    return tab, sel


def estabilidade(resultado) -> pd.DataFrame:
    raizes = resultado.roots
    mods = np.abs(raizes)
    return pd.DataFrame({"raiz (módulo)": mods, "estável (>1)": mods > 1})


# =============================================================================
# 4. CAUSALIDADE DE GRANGER (nos dois sentidos)
# =============================================================================


def causalidade_granger(dados: pd.DataFrame, p: int) -> pd.DataFrame:
    """
    Teste de causalidade de Granger, par a par, implementado manualmente via
    comparação de modelos OLS aninhados (F-test clássico de Granger, o mesmo
    princípio usado em Souza, Gil e Triches 2024): para cada par (causing,
    caused), estima-se a equação do VAR para `caused` com todas as defasagens
    das 4 endógenas (modelo irrestrito) e sem as defasagens de `causing`
    (modelo restrito), e testa-se se a diferença é conjuntamente significativa.

    Usado no lugar do `VARResults.test_causality` nativo do statsmodels: nesta
    versão da biblioteca (0.15.0) o teste nativo produziu estatísticas F
    negativas para alguns pares deste sistema — um resultado matematicamente
    impossível (F-stat é sempre ≥0) que indica um problema numérico na
    implementação nativa, não nos dados. A versão manual abaixo reproduz a
    mesma lógica de forma transparente e auditável.
    """
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
            linhas.append({
                "causa (Granger)": causing, "variável afetada": caused,
                "estatística F": F, "p-valor": pval,
                "significativo a 5%": bool(pval < .05),
            })
    return pd.DataFrame(linhas)


def whiteness_residuos(resultado, nlags=12) -> dict:
    teste = resultado.test_whiteness(nlags=nlags)
    return {"estatística (Portmanteau)": teste.test_statistic, "p-valor": teste.pvalue,
           "resíduos sem autocorrelação a 5%": teste.pvalue > .05}


# =============================================================================
# 5. IMPULSO-RESPOSTA E DECOMPOSIÇÃO DA VARIÂNCIA
# =============================================================================


def impulso_resposta(resultado, periodos=PERIODOS_IRF):
    irf = resultado.irf(periods=periodos)
    return irf


def extrair_irf_selic_para_ma(irf, alpha=0.05) -> pd.DataFrame:
    """Resposta de log(N_MA) a um choque ortogonalizado de 1 d.p. em ΔSelic,
    com intervalo de confiança (Monte Carlo / bootstrap dos coeficientes)."""
    i_selic = ENDOGENAS.index("dSelic")
    i_ma = ENDOGENAS.index("log_N_MA")
    resp = irf.orth_irfs[:, i_ma, i_selic]
    ic_baixo, ic_alto = irf.errband_mc(orth=True, repl=1000, signif=alpha)
    linhas = []
    for h in range(len(resp)):
        linhas.append({
            "horizonte (meses)": h,
            "resposta log(N_MA) a choque +1dp em ΔSelic": resp[h],
            "IC inferior": ic_baixo[h, i_ma, i_selic],
            "IC superior": ic_alto[h, i_ma, i_selic],
        })
    return pd.DataFrame(linhas)


def grafico_irf(irf, painel_path):
    fig = irf.plot(orth=True, signif=0.05, impulse="dSelic", response="log_N_MA")
    fig.set_size_inches(9, 5)
    fig.suptitle("Resposta de log(N_MA) a um choque ortogonalizado em ΔSelic (IC 95%)")
    fig.tight_layout()
    fig.savefig(painel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def decomposicao_variancia(resultado, periodos=PERIODOS_IRF) -> pd.DataFrame:
    fevd = resultado.fevd(periods=periodos)
    i_ma = ENDOGENAS.index("log_N_MA")
    tab = pd.DataFrame(fevd.decomp[i_ma], columns=ENDOGENAS)
    tab.index.name = "horizonte (meses)"
    return tab


# =============================================================================
# 6. ROBUSTEZ — SENSIBILIDADE DA IRF À ORDENAÇÃO DE CHOLESKY
# =============================================================================
#
# A IRF ortogonalizada decompõe a covariância contemporânea dos resíduos via
# Cholesky, o que exige assumir uma ordem entre as variáveis (quem reage a
# quem "no mesmo mês"). A ordem usada na seção principal (log_N_MA, dSelic,
# dIBC_Br, Ibov_Ret) assume que o M&A é a mais "exógena" contemporaneamente
# — decisão arbitrária, não teórica. Aqui testam-se 4 ordenações plausíveis
# para checar se a resposta de log(N_MA) a um choque de ΔSelic muda muito
# conforme a ordem escolhida. As defasagens (p) e os dados são os mesmos; só
# a ordem das colunas entra na estimação (o que só afeta a decomposição de
# Cholesky, não os coeficientes reduzidos do VAR).

ORDENACOES_TESTADAS = {
    "Original (M&A mais exógeno)":
        ["log_N_MA", "dSelic", "dIBC_Br", "Ibov_Ret"],
    "Selic mais exógena (padrão na literatura de choque monetário)":
        ["dSelic", "log_N_MA", "dIBC_Br", "Ibov_Ret"],
    "Atividade econômica primeiro (Selic reage à economia, M&A reage a ambas)":
        ["dIBC_Br", "dSelic", "log_N_MA", "Ibov_Ret"],
    "Mercado financeiro primeiro (Ibovespa mais exógeno)":
        ["Ibov_Ret", "dSelic", "dIBC_Br", "log_N_MA"],
}


def robustez_ordenacao(y: pd.DataFrame, exog: pd.DataFrame, p: int,
                       periodos=PERIODOS_IRF) -> pd.DataFrame:
    linhas = []
    for nome_ordem, ordem in ORDENACOES_TESTADAS.items():
        y_reordenado = y[ordem]
        modelo = VAR(y_reordenado, exog=exog)
        resultado = modelo.fit(p)
        irf = resultado.irf(periods=periodos)
        i_selic = ordem.index("dSelic")
        i_ma = ordem.index("log_N_MA")
        resp = irf.orth_irfs[:, i_ma, i_selic]
        for h, valor in enumerate(resp):
            linhas.append({"ordenação": nome_ordem, "horizonte (meses)": h,
                           "resposta log(N_MA) a choque +1dp em ΔSelic": valor})
    return pd.DataFrame(linhas)


def grafico_robustez_ordenacao(tab: pd.DataFrame, caminho):
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = ["#1f4e79", "#b7472a", "#2e7d32", "#6a3d9a"]
    for (nome_ordem, grupo), cor in zip(tab.groupby("ordenação", sort=False), cores):
        ax.plot(grupo["horizonte (meses)"], grupo["resposta log(N_MA) a choque +1dp em ΔSelic"],
               marker="o", markersize=3, label=nome_ordem, color=cor)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("horizonte (meses)")
    ax.set_ylabel("resposta de log(N_MA)")
    ax.set_title("Sensibilidade da IRF (ΔSelic → log(N_MA)) à ordenação de Cholesky")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resumo_robustez_ordenacao(tab: pd.DataFrame) -> pd.DataFrame:
    """Para cada horizonte, reporta o menor e o maior valor da IRF entre as
    4 ordenações, e se todas concordam no sinal — a checagem central de
    robustez: se o sinal muda conforme a ordenação, a IRF não é confiável
    para inferir a direção do efeito."""
    linhas = []
    for h, grupo in tab.groupby("horizonte (meses)"):
        valores = grupo["resposta log(N_MA) a choque +1dp em ΔSelic"]
        linhas.append({
            "horizonte (meses)": h, "mínimo": valores.min(), "máximo": valores.max(),
            "todas as ordenações concordam no sinal": bool((valores > 0).all() or (valores < 0).all()),
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 7. GRÁFICOS AUXILIARES
# =============================================================================


def grafico_series(painel: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    dados = {"log_N_MA": "log(N_MA)", "dSelic": "ΔSelic (p.p.)",
            "dIBC_Br": "ΔIBC-Br (%)", "Ibov_Ret": "Retorno Ibovespa (%)"}
    cores = ["#1f4e79", "#b7472a", "#6a3d9a", "#2e7d32"]
    for ax, (col, titulo), cor in zip(axes, dados.items(), cores):
        ax.plot(painel.index.to_timestamp(), painel[col], color=cor)
        ax.set_title(titulo)
        ax.axhline(0, color="k", lw=.5) if col != "log_N_MA" else None
        ax.axvspan(pd.Period(COVID_INICIO).to_timestamp(), pd.Period(COVID_FIM).to_timestamp(),
                  color="gray", alpha=.15)
    fig.tight_layout()
    fig.savefig(SAIDA / "series_var.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. EXECUÇÃO
# =============================================================================


def main():
    print("=" * 70)
    print("1. DADOS")
    bruto = carregar_negocios()
    negocios = aplicar_filtros(bruto)
    print(f"Base bruta: {len(bruto):,} operações | amostra final: {len(negocios):,}")

    painel = montar_painel(negocios)
    painel.to_csv(SAIDA / "painel_mensal_var.csv")
    grafico_series(painel)

    print("\n2. RAIZ UNITÁRIA DAS ENDÓGENAS DO SISTEMA VAR")
    ru = testes_raiz_unitaria(painel)
    ru.to_csv(SAIDA / "raiz_unitaria_var.csv", index=False)
    print(ru.round(3).to_string(index=False))
    if not ru["Estacionária a 5%"].all():
        print("-> ATENÇÃO: alguma endógena não estacionária permanece no sistema — "
              "ver seção 4 do texto de análise.")
    else:
        print("-> Todas as 4 endógenas passam no ADF a 5%: VAR em nível (sobre as "
              "variáveis já transformadas) é apropriado, sem precisar de VECM.")

    y, exog, dados = preparar_dados_var(painel)
    print(f"\nAmostra do VAR: n={len(dados)} meses "
          f"({dados.index.min()} a {dados.index.max()})")

    print("\n3. SELEÇÃO DO NÚMERO DE DEFASAGENS (até 12 meses)")
    tab_sel, sel = selecionar_defasagem(y, exog)
    tab_sel.to_csv(SAIDA / "selecao_defasagem.csv", index=False)
    print(tab_sel.to_string(index=False))
    p_aic, p_bic, p_hqic, p_fpe = sel.aic, sel.bic, sel.hqic, sel.fpe
    print(f"\nDefasagem sugerida — AIC: {p_aic} | BIC: {p_bic} | HQIC: {p_hqic} | FPE: {p_fpe}")
    p_escolhido = p_bic if p_bic and p_bic >= 1 else max(p_aic, 1)
    print(f"Defasagem adotada no VAR principal: p={p_escolhido} "
          f"(BIC, mais parcimonioso; AIC tende a superestimar p em amostras deste tamanho)")

    print(f"\n4. ESTIMAÇÃO DO VAR({p_escolhido}) — log(N_MA), ΔSelic, ΔIBC-Br, Ibov_Ret; "
          f"Covid como exógena")
    modelo = VAR(y, exog=exog)
    resultado = modelo.fit(p_escolhido)
    with open(SAIDA / f"var_p{p_escolhido}_sumario.txt", "w", encoding="utf-8") as f:
        f.write(str(resultado.summary()))
    print(f"AIC={resultado.aic:.3f}  BIC={resultado.bic:.3f}  HQIC={resultado.hqic:.3f}")

    print("\n5. ESTABILIDADE DO SISTEMA (raízes do polinômio característico)")
    est = estabilidade(resultado)
    est.to_csv(SAIDA / "estabilidade.csv", index=False)
    print(est.round(4).to_string(index=False))
    print("-> Sistema estável (todas as raízes fora do círculo unitário)."
          if est["estável (>1)"].all() else
          "-> ATENÇÃO: sistema pode ser instável — checar especificação.")

    print("\n6. AUTOCORRELAÇÃO DOS RESÍDUOS (teste de Portmanteau, 12 lags)")
    white = whiteness_residuos(resultado)
    pd.Series(white).to_csv(SAIDA / "whiteness_residuos.csv")
    print(white)

    print("\n7. CAUSALIDADE DE GRANGER (todos os pares, nos dois sentidos)")
    granger = causalidade_granger(dados, p_escolhido)
    granger.to_csv(SAIDA / "causalidade_granger.csv", index=False)
    print(granger.round(4).to_string(index=False))
    par = granger[(granger["causa (Granger)"] == "dSelic") & (granger["variável afetada"] == "log_N_MA")]
    par_inv = granger[(granger["causa (Granger)"] == "log_N_MA") & (granger["variável afetada"] == "dSelic")]
    print(f"\nΔSelic -> log(N_MA): p={par['p-valor'].iloc[0]:.4f} "
          f"({'significativo' if par['significativo a 5%'].iloc[0] else 'não significativo'} a 5%)")
    print(f"log(N_MA) -> ΔSelic: p={par_inv['p-valor'].iloc[0]:.4f} "
          f"({'significativo' if par_inv['significativo a 5%'].iloc[0] else 'não significativo'} a 5%)")

    print("\n8. FUNÇÃO DE IMPULSO-RESPOSTA — choque de 1 d.p. em ΔSelic sobre log(N_MA)")
    irf = impulso_resposta(resultado)
    tab_irf = extrair_irf_selic_para_ma(irf)
    tab_irf.to_csv(SAIDA / "irf_selic_para_ma.csv", index=False)
    print(tab_irf.round(5).to_string(index=False))
    grafico_irf(irf, SAIDA / "irf_dselic_para_logma.png")

    print("\n9. DECOMPOSIÇÃO DA VARIÂNCIA DO ERRO DE PREVISÃO DE log(N_MA)")
    fevd = decomposicao_variancia(resultado)
    fevd.to_csv(SAIDA / "fevd_log_ma.csv")
    print(fevd.round(4).to_string())
    print(f"\nNo horizonte de 12 meses, ΔSelic explica "
          f"{100*fevd.iloc[-1]['dSelic']:.1f}% da variância do erro de previsão de log(N_MA) "
          f"(o restante é a própria série + ΔIBC-Br + Ibov_Ret).")

    print("\n10. ROBUSTEZ — SENSIBILIDADE DA IRF À ORDENAÇÃO DE CHOLESKY")
    print(f"Testando {len(ORDENACOES_TESTADAS)} ordenações plausíveis das 4 endógenas "
          f"(mesma defasagem p={p_escolhido}, mesmos dados — só a ordem das colunas muda):")
    for nome_ordem, ordem in ORDENACOES_TESTADAS.items():
        print(f"  - {nome_ordem}: {' > '.join(ordem)}")
    tab_robustez = robustez_ordenacao(y, exog, p_escolhido)
    tab_robustez.to_csv(SAIDA / "robustez_ordenacao_cholesky.csv", index=False)
    grafico_robustez_ordenacao(tab_robustez, SAIDA / "robustez_ordenacao_cholesky.png")
    resumo = resumo_robustez_ordenacao(tab_robustez)
    resumo.to_csv(SAIDA / "robustez_ordenacao_resumo.csv", index=False)
    print(resumo.round(5).to_string(index=False))
    n_concordam = resumo["todas as ordenações concordam no sinal"].sum()
    print(f"\nEm {n_concordam} de {len(resumo)} horizontes, as 4 ordenações concordam no sinal "
          f"da resposta.")
    if n_concordam < len(resumo) * 0.5:
        print("-> A IRF é SENSÍVEL à ordenação escolhida: o sinal do efeito muda dependendo de "
              "qual variável se assume como mais exógena contemporaneamente. A leitura da seção "
              "8.3 (magnitude pequena e não significante em nenhum horizonte) já era cautelosa "
              "quanto a isso, e esta robustez reforça que a IRF não deve ser lida como uma "
              "conclusão forte sobre o sinal do efeito de curto prazo.")
    else:
        print("-> A IRF é relativamente ROBUSTA à ordenação: o sinal da resposta se mantém na "
              "maioria dos horizontes independentemente de qual variável é tratada como mais "
              "exógena contemporaneamente.")

    print(f"\nPronto. Resultados em: {SAIDA}")


if __name__ == "__main__":
    main()
