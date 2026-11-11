# Taxa Selic e fusões e aquisições no Brasil: evidências sobre o número de operações entre 2006 e 2024

Trabalho de Conclusão de Curso — Ciências Econômicas, Ibmec-DF, 2026
Autor: Alexandre de Sousa Campos Menezes · Orientador: Prof. MSc. Laerte Jun Takeuti

O trabalho estima, por Mínimos Quadrados Ordinários com erros-padrão de Newey-West, a associação entre o
nível da meta da taxa Selic (defasada em 1, 3, 6, 9 e 12 meses) e o número mensal de operações de M&A
anunciadas no Brasil entre janeiro de 2006 e dezembro de 2024 (228 meses, 2.225 operações).
Resultado principal: coeficiente de −0,0493 na defasagem de seis meses, cerca de 4,8% menos operações por
ponto percentual a mais na Selic.

## Estrutura

```
final/                         ARQUIVOS DEFINITIVOS (usados no TCC)
  codigo/tcc_mqo_final.py      script único: amostra, painel, testes, estimação, todas as tabelas e figuras
  dados/sgs/                   séries do BCB/SGS da execução final (432, 433, 24364, 3695)
  dados/macro_controls.csv     Ibovespa (B3) e IIE-Br (FGV IBRE), mensais
  dados/factset/               local da base da FactSet (NÃO publicada — ver LEIA-ME.md)
  resultados/                  saída da execução final: tab_2_1 … tab_6_8, quadro_2_1, fig_2_1 … fig_6_3,
                               painel_mensal.csv, fontes_dados.csv, execucao.json
referencias/                   textos de referência
arquivo/                       VERSÕES ANTIGAS (histórico; não usadas no trabalho final)
  codigo/  resultados/  analises/  dados/  tex/  plano_de_trabalho.md
```

Correspondência entre os arquivos de `final/resultados/` e o texto: `tab_X_Y_*.csv` é a Tabela X.Y,
`quadro_2_1_*.csv` é o Quadro 2.1 e `fig_X_Y_*.png` é a Figura X.Y.

## Como reproduzir

1. `pip install -r requirements.txt`
2. Copie a base da FactSet para `final/dados/factset/Deals_Results_TCC_FactSet_Att.xlsx`.
3. Na pasta `final/codigo`, rode `python tcc_mqo_final.py`.

Por padrão, o script usa as séries do Banco Central guardadas em `final/dados/sgs`, que reproduzem
exatamente os números do TCC. Com `python tcc_mqo_final.py --atualizar`, as séries são baixadas de novo da
API do BCB; os números podem mudar ligeiramente se o BCB tiver revisado alguma série.

## Dados e licença

A base de operações da FactSet foi obtida pela licença institucional do Ibmec-DF e é de uso restrito, por
isso não está no repositório. O repositório traz apenas dados públicos (BCB/SGS, B3, FGV IBRE) e resultados
agregados por mês.
