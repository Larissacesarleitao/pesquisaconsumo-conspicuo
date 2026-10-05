# Consumo conspícuo: associações com fatores socioeconômicos e redes sociais

Código e tabelas do Trabalho de Conclusão de Curso (MBA em Data Science e
Analytics, USP/Esalq), de Larissa Mendonça Cesar Leitão. 2026

## Sobre os dados
A base de respostas do questionário **não é disponibilizada**, por conter dados
individuais que poderiam permitir a reidentificação dos participantes, que foram
informados de que os dados seriam usados exclusivamente para fins acadêmicos.
**Estão publicados o código e as tabelas e figuras resultantes das análises.*

## Conteúdo
- `analise_consumo_conspicuo.py`: código principal. Lê a base, trata os dados e
  roda as análises do trabalho (perfil da amostra, construtos, validade,
  correlações e regressões).
- `tabelas_e_figuras.xlsx`: tabelas e figuras do trabalho. A numeração
  é a mesma do trabalho.
- `material_adicional/`: código das verificações de robustez com variáveis indicadoras
  (dummies), citadas na seção de Resultados.

## Como usar o código
O código espera a base em Excel na mesma pasta, aba "Base Resultado Final", com
as colunas na ordem do formulário original. Como a base não é pública, o código
serve para consulta e transparência dos procedimentos.

Pacotes usados: pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, openpyxl.

## Observações
- O alfa de Cronbach foi calculado no Excel.
- A AVE foi obtida por componentes principais, como aproximação (ver trabalho).
