# Plano de trabalho do TCC

## 1. Tema e pergunta de pesquisa

**Tema:** Taxa Selic e fusões e aquisições (M&A) no Brasil, entre 2005 e 2024.

**Pergunta:** Aumentos da Selic estão associados à redução posterior do número mensal de anúncios de fusões e aquisições no Brasil?

## 2. Objetivo

Analisar a relação entre a Selic e os anúncios de M&A, considerando a persistência das operações, o mercado de ações e o período da pandemia.

**Hipótese principal:** aumentos da Selic são seguidos por uma redução no número de anúncios de M&A.

## 3. Dados

Construir uma base mensal de janeiro de 2005 a dezembro de 2024, conforme a disponibilidade das séries:

## 4. Método

O modelo terá como base metodológica **Grönroos (2024)**, que utiliza regressão linear múltipla por MQO para analisar a atividade de M&A, com indicadores macroeconômicos contemporâneos ou defasados em um mês e uma dummy da Covid-19

```text
M&A_t = α + β1 M&A_(t−1) + β2 Selic_(t−k)
          + β3 RetornoIbov_t + β4 Covid_t + ε_t
```

Onde:

- `M&A_t`: quantidade de anúncios de operações no mês atual.
- `M&A_(t−1)`: quantidade de anúncios no mês anterior.
- `Selic_(t−k)`: Selic defasada em `k` meses.
- `RetornoIbov_t`: retorno mensal do Ibovespa no mês atual.
- `Covid_t`: dummy igual a 1 de março a dezembro de 2020 e 0 nos demais meses. Esse recorte representa o choque inicial, e não toda a duração da pandemia.
- `α`: constante; `β1` a `β4`: coeficientes estimados.
- `ε_t`: erro do modelo.

### Defasagens da Selic

Estimar três regressões separadas, mantendo os mesmos controles:

- **Modelo 1:** Selic com defasagem de 1 mês.
- **Modelo 2:** Selic com defasagem de 3 meses.
- **Modelo 3:** Selic com defasagem de 6 meses.

Não incluir inicialmente as três defasagens juntas, pois elas podem apresentar elevada correlação entre si, dificultando a estimação dos efeitos individuais (multicolinearidade). Usar a mesma amostra de meses nos três modelos para facilitar a comparação.

### Análise dos resultados

1. Apresentar gráficos e estatísticas descritivas.
2. Verificar a estacionariedade das séries com o teste ADF; se necessário, transformar as variáveis e ajustar a interpretação da equação.
3. Estimar os três modelos e verificar a autocorrelação dos resíduos.
4. Comparar o sinal, a magnitude e os intervalos de confiança do coeficiente da Selic. Verificar também a sensibilidade ao período escolhido para a dummy.



## 5. Referências de apoio

- **GRÖNROOS, Otto. _European M&A Activity: Influences of Macroeconomic Indicators and Covid-19_. 2024. Dissertação de mestrado — Hanken School of Economics, Helsinki, 2024.** Referência metodológica principal para a regressão múltipla por MQO, o uso de indicadores defasados e a dummy da pandemia. [Consultar PDF](data/Gronroos_2024_European_MA.pdf).
- **Batista et al. (2022), _Mergers and acquisitions in Brazil and economic uncertainty shocks_:** apoio à escolha dos controles macroeconômicos e à discussão da incerteza econômica;
- **Souza, Gil e Triches (2024), _Economic Performance: An Analysis of Mergers and Acquisitions in Brazil_:** apoio para estudar Selic, atividade econômica e precedência temporal, com comparação entre operações domésticas e transfronteiriças.
- **Gonzalez (2024), _The influence of interest & inflation rate on M&A processes_:** apoio à discussão sobre juros, inflação e diferenças entre quantidade e valor das transações.
