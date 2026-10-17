# Plano de trabalho do TCC

Atualizado em 17/10/2026.

## 1. Objetivo

Analisar a associação entre a Selic e a quantidade mensal de anúncios de fusões e aquisições no Brasil, entre 2005 e 2024, conforme os filtros da base.

## 2. Histórico do trabalho

- **Referências:** Revisão dos estudos sobre Selic e fusões e aquisições.
- **Método adotado:** Regressão linear mensal por MQO, adaptada de Grönroos (2024), com erros-padrão robustos HAC de Newey-West.
- **Preparação e análise:** Organização da base mensal, estimação das regressões e realização dos testes.
- **Revisão do código:** Identificação do erro nas defasagens e recálculo dos resultados.
- **Atualização do código:** Correção das defasagens nos novos scripts, adoção da meta Selic do Banco Central e inclusão de atividade econômica, sazonalidade e Stringency.
- **Comparação atual:** O nível transformado da meta Selic apresentou associação negativa com M&A. O modelo de nove meses teve o melhor ajuste; seis meses foi recomendado como principal.
- **Situação atual:** Tabelas e gráficos foram produzidos, mas ainda é necessário revisar a base e consolidar código, resultados e texto em uma execução final.

## 3. Modelo definido para a conclusão do TCC

- **Método:** regressão linear estimada por MQO, com erros-padrão robustos HAC de Newey-West.
- **Variável dependente:** quantidade mensal de operações em logaritmo.
- **Variável principal:** nível transformado da meta Selic.
- **Modelo principal:** defasagem de seis meses.
- **Modelo alternativo:** nove meses, que apresentou melhor ajuste estatístico.




## 4. Variáveis e justificativas

- **justificar** cada escolha das variaveis na metodologia. Não retirar variáveis apenas por falta de significância.

- **Conferir as duplicidades:** verificar na fonte os IDs `706879MM` e `734331MM` e manter uma ocorrência de cada duplicação confirmada. A amostra deverá passar de 2.292 para 2.290 operações. Não excluir automaticamente outros registros semelhantes.

- **Explicar o limite da previsão:** o modelo usa o Ibovespa e o IBC-Br do próprio mês para estimar as operações daquele mês. Esses dados não eram conhecidos no início do mês. Portanto, apresentar o exercício como uma avaliação do modelo com dados já conhecidos, e não como uma previsão feita antecipadamente.

- **justificar:** manter seis meses como referência por integrar o desenho inicial, representar um intervalo economicamente plausível e apresentar resultados robustos e diagnósticos aceitáveis, com as limitações registradas no trabalho. O modelo de nove meses será apresentado como alternativa, pois obteve o melhor ajuste estatístico. Os diagnósticos devem ser conferidos novamente após as correções.


## 6. Referências e materiais de apoio

- **Grönroos (2024):** referência principal para o método.
- **Batista et al. (2022):** fatores macroeconômicos e incerteza.
- **Souza, Gil e Triches (2024):** Selic e M&A no Brasil.
- **Gonzalez (2024):** juros, inflação e M&A.
