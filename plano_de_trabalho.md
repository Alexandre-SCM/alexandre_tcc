# Plano de trabalho do TCC

## 1. Objetivo

Analisar a relação entre a Selic e a quantidade mensal de anúncios de fusões e aquisições no Brasil, entre 2005 e 2024.

## 2. Método em análise

- Regressão linear múltipla por MQO, adaptada de Grönroos (2024).
- Quantidade de operações em logaritmo e variação da Selic defasada em 1, 3 e 6 meses.
- Controles: operações do mês anterior, retorno do Ibovespa e período da pandemia.

## 3. O que já foi feito

- Consulta às referências e definição da proposta de pesquisa.
- Organização da base mensal e estimação de diferentes modelos.
- Comparação das regressões e da significância das variáveis.
- Verificação dos principais testes e da sensibilidade dos resultados.
- Identificação de erro nas defasagens e recálculo em uma análise separada; os scripts originais ainda precisam ser corrigidos.

## 4. Resultado preliminar

- O modelo mensal corrigido de um mês apresentou o melhor ajuste entre as três defasagens analisadas.
- A variação da Selic apresentou associação negativa e significativa na especificação principal.
- As operações do mês anterior foram significativas; Ibovespa e a dummy inicial da pandemia não foram.
- O resultado da Selic depende dos controles. Ainda há autocorrelação nos resíduos e sazonalidade a tratar antes de definir o modelo final.

## 5. Próximos passos

1. Conferir a base original, os filtros das operações e a medida da Selic utilizada.
2. Corrigir as defasagens nos scripts e atualizar os resultados.
3. Ajustar o modelo para tratar sazonalidade e dependência temporal; repetir os testes.
4. Definir o modelo final com justificativa teórica e estatística.
5. Preparar as tabelas e os gráficos finais.


## 6. Materiais de apoio

- **Grönroos (2024):** referência principal para o método.
- **Batista et al. (2022):** fatores macroeconômicos e incerteza.
- **Souza, Gil e Triches (2024):** Selic e M&A no Brasil.
- **Gonzalez (2024):** juros, inflação e M&A.
- [Análise detalhada dos modelos mensais](src/analise_mensal/analise_mensal.md).
