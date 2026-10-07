# Buscador — Goodreads  para Genêros Romance + Poetry

Projeto 1 da disciplina de Recuperação de Informação e Sistemas de Recomendação.

Esse domínio foi escolhido pois, além de atender aos pré-requisitos do projeto (buscador + recomendador) é um tema de interesse meu. 

## 1. Domínio e corpus

- **Domínio:** Goodreads, gêneros Romance e Poetry.

- **Corpus indexado:** descrições dos livros (`description`) de
  `goodreads_books_romance.json.gz` e `goodreads_books_poetry.json.gz` (UCSD Book Graph).

- **Estrutura usuário-item confirmada:** A estrutura usuário-item foi confirmada nos arquivos `goodreads_interactions_romance.json.gz` e `goodreads_interactions_poetry.json.gz` (UCSD Book Graph, filtrados por gênero). O conjunto de interações reúne 698.971 usuários únicos e 371.794 livros únicos, totalizando 45.527.206 interações (rating explícito de 0 a 5, ou marcação de leitura), com densidade de 0,0175% (esparsidade de 99,98%). A divisão por gênero é de 42.792.856 interações em romance (655.454 usuários, 335.447 livros) contra 2.734.350 em poetry (377.799 usuários, 36.514 livros) — romance responde por cerca de 94% do volume de interações, uma assimetria relevante para o desenho do Recomendador. O cold-start é acentuado: 11,8% dos usuários e 5,1% dos livros têm apenas 1 interação; 50,9% dos usuários e 32,4% dos livros têm 10 ou menos.

- **Tamanho final do corpus indexado:** O corpus final, após filtro de idioma (inglês) e remoção de duplicatas entre gêneros, contém 295.311 documentos: 272.247 de romance e 23.064 de poetry (92,2% / 7,8%). O vocabulário do corpus é de 632.305 termos únicos (contagem bruta, por split()), com tamanho médio de descrição de 156,4 palavras.

## 2. Ambiente e setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Baixe os arquivos do corpus em : 
Fonte: [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/)

## 3. Estrutura do repositório

```
projeto-buscador/
  data/                    # raw/ e processed/, nunca versionados
  src/
    preprocessing.py       # tokenização, stopwords, stemming
    data_loading.py        # carregamento do corpus via DuckDB
    indexing.py             # índice invertido
    models/
      boolean.py
      vetorial.py            # TF-IDF via sklearn
      probabilistico.py      # BM25
    evaluation.py            # pooling e métricas
  notebooks/
    explore_vocabulary.py    # exploração de vocabulário real do corpus
    check_final_queries.py    # validação das consultas utilizadas
  tests/
    test_preprocessing.py
    test_models.py
  explorar_corpus.py         # primeira exploração (Ambientação Técnica)
  main.py                    # orquestra o pipeline completo
  requirements.txt
  README.md
```

## 4. Pipeline de pré-processamento

| Etapa | Decisão | Justificativa |
|---|---|---|
| Tokenização | `nltk.word_tokenize` | A tokenização com NLTK lida com as ambiguidades da língua humana, como pontuações coladas em palavras e abreviações e  diferente do .split() do Python, ela é capaz de isolar símbolos mantendo o contexto gramatical do texto. |
| Case folding | Lowercase | Padrão, reduz vocabulário sem perda de recall relevante |
| Stopwords | NLTK inglês, exceto palavras de negação | Negação muda o sentido de descrições de romance ("not a typical love story") |
| Stemming vs. Lematização | Stemming (Snowball, inglês) | Devido ao tamanho do corpus trabalhado, a lematização seria menos eficiente que o stemming, em termos de processamento e tempo de execução. |
|Expansão de contrações em inglês| Substituição |Para evitar que termos como 'isn't' se transformem em 'isn' + 't', foi aplicado esse tratamento para padronizar o texto limpando as abreviações negativas |

**Verificação de idioma:** : Puxando uma amostra dos registros, foi possível observar a predominância de documentos em inglês. No entanto, durante a primeira execução do arquivo 'explore_vocaulary' foi observado na saída termos como ,'la', 'el' entre outros, sugerindo a existência de uma amostra de documentos em outro idioma. Para evitar ruídos, foi aplicado um filtro de idioma durante a etapa de carregamento dos dados juntamente a uma deduplicação para garantir apenas uma ocorrência de cada obra. 

## 5. Modelos de ranking implementados

Conforme exigido foram implementados os modelos **Booleano**, **Vetorial** e **Probabilístico**, eles foram comparados sobre as mesmas consultas.

### 5.1 Booleano (`src/models/boolean.py`)

O modelo Booleano apresentou o desempenho mais baixo entres os modelos aplicados, obtendo uma precisão média de apenas 5% indica que, quando documentos são recuperados, a maioria deles não é considerada relevante segundo as marcações utilizadas no conjunto de avaliação. Já o recall médio de 6,8% também demonstrou uma baixa capacidade de recuperar os documentos considerados relevantes.

Esse comportamento está ligado a uma limitação deste modelo, que é agravado considerando o tipo de consulta utilizado. Quando buscamos consultas para um subgênero literário, nem sempre as descrições vão apresentar de forma explícita as palavras contidas na consulta, e sim conceitos, características da narrativa ou até mesmo relações entre elementos. Por exemplo, um livro pode apresentar uma relação de friends to lovers sem utilizar exatamente essa expressão em sua descrição.

Dessa forma, a estratégia booleana tende a ser muito rígida para esse conjunto de dados. A presença ou ausência dos termos não é suficiente para representar adequadamente a relevância definida pelo julgamento manual.


### 5.2 Vetorial (`src/models/vetorial.py`)

TF-IDF via `sklearn.feature_extraction.text.TfidfVectorizer` + similaridade de cosseno.
Reaproveita a mesma função de pré-processamento usada no índice, para manter o
vocabulário consistente entre os três modelos.

Já este modelo apresentou um desempleno razoavel, com uma precisão média de 0,370, recall de 0,267 e F1 de 0,293, ele superou significativamente o Booleano já que supera sua rigidez, ao considerar a similaridade entre documentos e consultas ao inves de somente a correspondencia. No entanto, para consultas especificas como contemporary romance alcançou precisão de 0,9  mas  um recall de apenas 0,391. O mesmo comportamento aparece em small town romance, com precisão de 0,9 e recall de 0,429. Ou seja, esse modelo consegue encontrar documentos relacionados com a cosulta, mas não recupera uma quantidade considerável dos documentos que foram classificados como relevantes. 

Logo, o TF-IDF possui limitações quando a relevância depende da interpretação de conceitos ou relações narrativas.

### 5.3 Probabilístico (`src/models/probabilistico.py`)
BM25 (k1 = 1.5, b = 0.75, valores padrão da literatura). 

O BM25 apresentou os melhores resultados gerais.

Sua precisão média foi de 0,590 (cerca de 20 pontos percentuais acima do TF-IDF) e seu recall de 0,515 foi quase o dobro do TF-IDF. Além disso, o  BM25 apresentou F1 superior ao TF-IDF em 7 das 10 consultas, enquanto o TF-IDF foi superior em apenas uma, sonnet collection classic. Para as consultas contemporary romance e small town romance, houve empate.

### Comparação entre os modelos


De forma geral, os resultados indicam uma hierarquia clara entre os modelos:
BM25 > TF-IDF > Booleano

O modelo Booleano apresentou desempenho muito baixo, demonstrando que a correspondência rígida de termos não é uma opção viável para o conjunto de consultas utilizado. Já o TF-IDF apresentou uma melhoria substancial, especialmente em consultas com maior correspondência lexical, mas apresentou recall relativamente baixo. Para finalizar, o BM25 apresentou o melhor equilíbrio entre precision e recall.

No entanto, tanto o BM25 quanto o TF-IDF demostraram uma perda de performance frente a consultas que exigem a identificação de relações narrativas ou combinação de múltiplos critérios, como enemies to lovers, slow burn romance e free verse nature poetry. Iss ose deve ao fato se serem limitados pela natureza predominantemente lexical da recuperação.

Assim, os resultados sugerem que o BM25 é a abordagem mais adequada entre as três avaliadas para o conjunto de dados deste experimento, mas também evidenciam uma possível limitação dos modelos tradicionais de recuperação de informação quando a definição de relevância depende de características semânticas que não estão necessariamente expressas pelos mesmos termos utilizados na consulta.


## 6. Avaliação experimental

### 6.1 Consultas de teste

As consultas utilizadas para o teste foram :

["second chance love story", "contemporary romance", "enemies to lovers", "grief and loss poems",
 "love poems about heartbreak", "free verse nature poetry", "small town romance", "slow burn romance",
 "sonnet collection classic", "friends to lovers"]

 Para validar essas consultas, foi utilizado o script notebooks/check_final_queries.py para garantir que o número de documentos recuperados via AND booleano não era muito baixo e nem excessivo.

### 6.2 Metodologia de julgamento de relevância
- **Pooling**: top-10 de cada um dos três modelos por consulta, unidos,
  embaralhados e julgados **sem saber qual modelo gerou cada resultado**
  (`ri_outputs/judgment_pool.csv`).
- Julgado por: O resultado do pool foi avalidado manualmente, para validar sua assertividade.
- **Limitação a declarar:** Recall é aproximado sobre o pool julgado, não
  sobre o corpus inteiro — documentos relevantes fora do top-10 de todos os
  modelos não entram no denominador.

### 6.3 Resultados

Abaixo esta a tabela com o resumo das métricas para cada modelo considerado:

| Modelo | Precision | Recall | F1 |
|---|---:|---:|---:|
| Booleano | **0,050** | **0,068** | **0,055** |
| TF-IDF | **0,370** | **0,267** | **0,293** |
| BM25 | **0,590** | **0,515** | **0,510** |

### 6.4 Discussão e limitações

- As limitações de cada modelo já foram discutidas na Seção 5.
- Validação Única: O julgamento de relevância foi realizado por um único anotador humano, o que introduz subjetividade na definição de relevância.
- Tamanho do Pool: Avaliou-se apenas o top-10 de cada modelo, o que restringe a estimativa do Recall real do sistema no acervo de 295 mil livros.
- Assimetria do Corpus: O corpus possui forte desbalanceamento (92,2% Romance vs 7,8% Poesia), refletindo na quantidade de documentos recuperados por tema.


## 7. Como rodar

```bash
python explorar_corpus.py              # exploração básica (Ambientação Técnica)
python notebooks/explore_vocabulary.py # vocabulário real, para calibrar as queries
python main.py                          # 1a vez: gera o pool de julgamento
# --> preencha ri_outputs/judgment_pool.csv manualmente (coluna relevant_0_ou_1)
python main.py                          # 2a vez: calcula as métricas
pytest tests/                           # roda os testes
```

## 8. Uso de IA neste projeto

Uso de IA, para validação de ulguns códigos e da criação do set-up inicial do projeto. Tambem para contornar problemas como o estouro de memória durante o carregamento das bases.
