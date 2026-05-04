# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Relatório de Análise de Dominio
#
# * O objetivo desse notebook é analisar os resultados obtidos no nosso banco de dados
# e estudar o comportamento deles para obter informações como as seguintes:
# Os slugs estão sendo muito importantes para o ranqueamento da empresa mencionada
# dentro das perguntas? Se sim, quais são os termos mais comuns usados para obter esse
# resultado?
# Existe uma correlação entre os termos Query e URL_Hostname? Se sim,
# que tipo de query ou URL pode causar que tenham essa relação?

# %% [markdown]
# Para esse estudo, foram usados dados das tabelas de **prompt_monitoring**,
# **prompt_searches** e **prompt_monitoring** e o notebook é adaptado para mudar
# facilmente o nosso alvo de estudo, para que podamos analisar novamente no futuro,
# entre mais dados, melhor.

# %% [markdown]
# O código para definir nosso estudo será definido no começo para manter simplicidade

# %% [markdown]
# Imports

# %%
import ast
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from scipy.stats import gaussian_kde
from unidecode import unidecode

# %%
#path1 = '/home/daniel/code/braviumx/data/domain_analysis_searches_and_monitoring_v1.csv'
#path2 = '/home/daniel/code/braviumx/data/search_and_cit_union_v1.csv'
#path3 = '/home/daniel/code/braviumx/data/query_fan_out.csv'

path1 = '/home/endless_light/cite-prob-llm/notebook_data/prompt_mon_and_search_new.csv'
path2 = '/home/endless_light/cite-prob-llm/notebook_data/search_and_cit_union_v1.csv'
path3 = '/home/endless_light/cite-prob-llm/notebook_data/query_fan_out.csv'

df1 = pd.read_csv(path1)
df1_name = "Domain Analysis - Searches and Monitoring"
df2 = pd.read_csv(path2)
df2_name = "Domain Analysis - Prompt and Searches"
df3 = pd.read_csv(path3)
df3_name = "Domain Analysis - Prompt and Query Fan Out"

# %%
df2.columns.values[2] = 'query' 
df3.columns.values[4] = 'query' 

# %% [markdown]
# ### Criação dos arquivos a serem estudados

# %% [markdown]
# Começaremos com a criação dos arquivos a serem estudados,
# fazendo um tratamento de stopwords, normalização dos dados,
# cálculos para contagem de palavras que dão "match" entre query e URL
# e também porcentagem de similaridade baseado no método de Jaccard.

# %% [markdown]
# Definimos o nome do output dos arquivos no seguinte bloco

# %%
output_1 = '/home/endless_light/cite-prob-llm/notebook_output/domain_analysis_results_searches_and_monitoring_v1.csv'
output_2 = '/home/endless_light/cite-prob-llm/notebook_output/domain_analysis_results_search_and_cit_union_v1.csv'
output_3 = '/home/endless_light/cite-prob-llm/notebook_output/domain_analysis_results_prompt_query_fan_out.csv'

# %% [markdown]
# Definindo as colunas a serem estudadas

# %%
col_query = "query"
col_url = "url"
col_hostname = "url_hostname"
prompt_query_fan_out = "prompt_query_fan_out"

# %%
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
nltk.download('punkt', quiet=True)


# %% [markdown]
# ###

# %%
def calculate_similarity(text1:str, text2:str) -> pd.Series:
    """
    Calculate similarity between two texts while ignoring stopwords and cleaning URLs.
    Based on Jaccard similarity.
    """

    if not isinstance(text1, str):
        text1 = ""
    if not isinstance(text2, str):
        text2 = ""

    stopwords_list = set(stopwords.words('portuguese'))
    stopwords_url = {'www', 'http', 'https', 'com', 'br', 'org', 'net', 'html', 'php'}
    all_stopwords = {unidecode(w) for w in stopwords_list.union(stopwords_url)}

    def preprocess(text:str) -> set:
        table = re.sub(r'[\W_]+', ' ', text)
        text_no_punct = unidecode(table.lower())
        tokens = text_no_punct.split()
        return {word for word in tokens if word not in all_stopwords}

    set1 = preprocess(text1)
    set2 = preprocess(text2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    match_count = len(intersection)

    if len(union) > 0:
        score = (match_count / len(union)) * 100
    else:
        score = 0.0

    return pd.Series([match_count, score, list(intersection)])


def run_analysis(df, col_query, col_target, suffix_name) -> pd.DataFrame: #type: ignore
    """
    Run function calculate_similarity and create columns with specific names.
    suffix_name: serves to differentiate 'url' and 'url_hostname'
    """

    new_col_matches = f'matches_{suffix_name}'
    new_col_score = f'score_{suffix_name}'
    new_col_words = f'common_words_{suffix_name}'

    df[[new_col_matches, new_col_score, new_col_words]] = df.apply(
        lambda row: calculate_similarity(row[col_query], row[col_target]),
        axis=1
    )

    return df


try:
    df1 = run_analysis(df1, col_query, col_url, suffix_name = "url")
    df1 = run_analysis(df1, col_query, col_hostname, suffix_name = "url_hostname")

    path_output1 = output_1
    df1.to_csv(path_output1, index=False)

except FileNotFoundError:
    print("Error: File not found. \n")


try:
    df2 = run_analysis(df2, col_query, col_url, suffix_name = "url")
    df2 = run_analysis(df2, col_query, col_hostname, suffix_name = "url_hostname")

    path_output2 = output_2
    df2.to_csv(path_output2, index=False)

except FileNotFoundError:
    print("Error: File not found. \n")



try:
    df3 = run_analysis(df3, prompt_query_fan_out, col_url, suffix_name = "url")
    df3 = run_analysis(df3, prompt_query_fan_out, col_hostname, suffix_name = "url_hostname")

    path_output3 = output_3
    df3.to_csv(path_output3, index=False)

except FileNotFoundError:
    print("Error: File not found. \n")

# %% [markdown]
# A seguir vamos a fazer nossas análises, respondendo uma pergunta dos objetivos
# por vez e finalmente tirando uma conclusão dos resultados

# %%
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


df1 = pd.read_csv(output_1)
df2 = pd.read_csv(output_2)
df3 = pd.read_csv(output_3)

col_lists = ['common_words_url', 'common_words_url_hostname']
for df in [df1, df2, df3]:
    for col in col_lists:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])



# %% [markdown]
# ## 0. Testes preliminares

# %% [markdown]
# ### 0.1 Fan-out

# %%
def fan_out_diagnosis(df, col_father = 'monitoring_id') -> None: #type: ignore
    n_total_lines = len(df)
    n_unique_fathers = df[col_father].nunique()

    ratio = n_total_lines / n_unique_fathers

    if ratio > 1.0:
        print(
            f"Fan-out detected: {
                n_total_lines
                } total lines for {n_unique_fathers} unique {col_father}s (ratio: {ratio:.2f})"
                )
    else:
        print(
            f"No fan-out: {
                n_total_lines
                } total lines for {n_unique_fathers} unique {col_father}s (ratio: {ratio:.2f})"
                )

fan_out_diagnosis(df1, "prompt_hash")
fan_out_diagnosis(df2, "prompt_hash")
fan_out_diagnosis(df3, "prompt_hash")


# %% [markdown]
# ### 0.2 Amostra detalhada

# %%
def analisar_volume_por_marca(
        df, dataset_name, #type: ignore
        col_brand='brand_name', #type: ignore
        col_id='prompt_hash', #type: ignore
        top_n = 20) -> None: #type: ignore
    """
    Analisa o 'Share of Voice' de cada marca dentro do dataset.
    Compara o volume total de linhas vs. prompts únicos.
    """
    print(f"--- Análise de Representatividade por Marca: {dataset_name} ---")

    stats = df.groupby(col_brand)[col_id].agg(['count', 'nunique'])


    stats.columns = ['Total Lines (Volume)', 'Unique Prompts']

    stats['Avg Rows/Prompt'] = stats['Total Lines (Volume)'] / stats['Unique Prompts']

    total_dataset = len(df)
    stats['% of Dataset'] = (stats['Total Lines (Volume)'] / total_dataset) * 100

    stats = stats.sort_values('Total Lines (Volume)', ascending=False)

    print(stats)
    plot_data = stats.head(top_n)

    plt.figure(figsize=(12, 6))

    ax = sns.barplot(x=plot_data['Total Lines (Volume)'], y=plot_data.index, palette="viridis")

    plt.title(f"Volume Total de Dados por Marca (Peso no Dataset) - {dataset_name}")
    plt.xlabel("Total de Linhas (Buscas/Citações)")
    plt.ylabel("Marca")

    max_x = stats['Total Lines (Volume)'].max()
    plt.xlim(0, max_x * 1.15)

    for i, (vol, pct) in enumerate(
        zip(plot_data['Total Lines (Volume)'], plot_data['% of Dataset'],
            strict=False)
        ):
        label = f"{int(vol)} ({pct:.1f}%)"
        ax.text(vol + (max_x * 0.01), i, label, va='center', fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

try:
    analisar_volume_por_marca(df1,df1_name, col_brand='brand_name', top_n=20)
    analisar_volume_por_marca(df2,df2_name, col_brand='brand_name', top_n=20)
    analisar_volume_por_marca(df3,df3_name, col_brand='brand_name', top_n=20)
except KeyError as e:
    print(f"Erro de coluna: {e}")


# %% [markdown]
# ## 1. Existe correlação entre Query e URL?
#
#

# %% [markdown]
# ### 1. Análise Similiridade de Jaccard
#
# Aqui veremos a distribuição dos scores, se a correlação for alta
# veremos "morros" mais na direita do gráfico, se for baixa, estarão na esquerda.

# %%
def compare_scores(df1, col_name1, df2, col_name2, df3, col_name3, col_score) -> None: #type: ignore
    n1 = len(df1)
    n2 = len(df2)
    n3 = len(df3)

    plt.figure(figsize=(10, 6))

    label_df1 = f"{col_name1} (n={n1})"
    label_df2 = f"{col_name2} (n={n2})"
    label_df3 = f"{col_name3} (n={n3})"

    sns.kdeplot(df1[col_score], label=label_df1, fill=True, color = "skyblue", clip=(0, 100))
    sns.kdeplot(df2[col_score], label=label_df2, fill=True, color = "orange", clip=(0, 100))
    sns.kdeplot(df3[col_score], label=label_df3, fill=True, color = "green", clip=(0, 100))

    plt.title(f"Comparação de Similaridade: Query vs {col_score}")
    plt.xlabel("Score de Similaridade (0 a 100%)")
    plt.ylabel("Densidade")
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()


compare_scores(df1, df1_name, df2, df2_name, df3, df3_name,"score_url")


# %% [markdown]
# O gráfico acima valida que existe uma correlação, no entanto, parece fraca visivelmente,
# isto é provavelmente devido a que o nosso método usado penaliza URL longas, já que é
# comparado qual o % de palavras que compõem essa URL aparece de fato no match.
# Como próximo passo, faremos uma análise em relação ao tamanho da query para verificar
# que o nosso método "pessimista" de fato está jogando as respostas para baixo.
#

# %% [markdown]
# ### 2. Análise de cobertura da query

# %% [markdown]
# Para corroborar nossos dados iremos agora criar um gráfico de cobertura
# que reflete em somente quanto da query está contida na URL

# %%
def calculate_coverage_query(df, col_matches='common_words_url', col_query=col_query) -> pd.Series: #type: ignore
    """
    Creates column based on % of words in the query that are found in the URL.
    Denominator is total words in the query after stopword removal.
    Numerator is total matched words in the URL.
    """

    stopwords_list = set(stopwords.words('portuguese'))
    stopwords_url = {'www', 'http', 'https', 'com', 'br', 'org', 'net', 'html', 'php'}
    all_stopwords = {unidecode(w) for w in stopwords_list.union(stopwords_url)}

    def preprocess(row) -> float: #type: ignore
        raw_text = str(row[col_query])
        table = re.sub(r'[\W_]+', ' ', raw_text)
        text_no_punct = unidecode(table.lower())
        tokens = [p for p in text_no_punct.split() if p not in all_stopwords and len(p) > 2]
        total_query_words = len(tokens)
        total_matches = len(row[col_matches])

        if total_query_words == 0:
            return 0.0
        pct = (total_matches / total_query_words) * 100
        return min(pct, 100.0)

    return df.apply(preprocess, axis=1)

df1['coverage_score'] = calculate_coverage_query(df1)
df2['coverage_score'] = calculate_coverage_query(df2)
df3['coverage_score'] = calculate_coverage_query(
    df3,
    col_matches='common_words_url',
    col_query=prompt_query_fan_out
    )


# %%
def compare_coverage(df1, col_name1, df2, col_name2, df3, col_name3, col_coverage) -> None: #type: ignore
    n1 = len(df1)
    n2 = len(df2)
    n3 = len(df3)

    plt.figure(figsize=(10, 6))

    label_df1 = f"{col_name1} (n={n1})"
    label_df2 = f"{col_name2} (n={n2})"
    label_df3 = f"{col_name3} (n={n3})"

    sns.kdeplot(df1[col_coverage], label=label_df1, fill=True, color = "skyblue")
    sns.kdeplot(df2[col_coverage], label=label_df2, fill=True, color = "orange")
    sns.kdeplot(df3[col_coverage], label=label_df3, fill=True, color = "green")

    plt.title(f"Comparação de Cobertura: Query vs {col_coverage}")
    plt.xlabel("Cobertura da Query (%)")
    plt.ylabel("Densidade")
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



compare_coverage(
    df1, df1_name,
    df2, df2_name,
    df3, df3_name,
    "coverage_score"
)


# %% [markdown]
# Com o gráfico acima podemos ver que de fato estavamos punindo muito os textos longos,
# já que conseguimos ver com mais clareza aqui que existem uma cobertura maior do que
# esperado na query, no entanto, a pesar de ter uma correlação clara, também vemos como
# existem muitos casos onde ele opera de forma semântica, dando a entender que o conteúdo
# dentro do site também é interessante para a busca. Podemos ver também como existe uma
# lacuna que contém um mínimo em um lugar entre 0 e 10, queremos analisar isto com mais
# detalhe pelo qual irei calcular onde ele existe e depois criar 2 grupos de estudo,
# 1 grupo contendo valores 0 e x e outro para valores mayores que x, onde x é dito mínimo.

# %% [markdown]
# A seguir iremos fazer uma boostrap analysis

# %%
def bootstrap_charts(df1, name1, df2, name2, df3, name3, col_target, n_boot = 2000) -> None: #type: ignore
    """
    Faz boostrap das distribuições e plota histogramas das médias amostrais individuais
    e em conjunto para melhorar a análise.
    """

    configs = [
        (df1, name1, 'skyblue'),
        (df2, name2, 'orange'),
        (df3, name3, 'green')
    ]

    stored_results = {}

    for df, name, color in configs:
        values = df[col_target].dropna().values
        n_size = len(values)

        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(values, size=n_size, replace=True)
            boot_means.append(np.mean(sample))

        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        mean_est = np.mean(boot_means)

        stored_results[name] = {
            'means': boot_means,
            'color': color,
            'stats': (mean_est, ci_lower, ci_upper)
        }

        plt.figure(figsize=(12, 6))

        sns.histplot(boot_means,
                     color = color,
                     kde=True,
                     element='step',
                     stat='density',
                     label = f"Distribuição das Médias - {name}")

        plt.axvline(mean_est, color="black", linestyle="--", #type: ignore
                    linewidth=2, label=f"Mean Estimate: {mean_est:.2f}")
        plt.axvline(ci_lower, color="red", linestyle="--", #type: ignore
                    linewidth=2, label=f"95% CI Lower: {ci_lower:.2f}")
        plt.axvline(ci_upper, color="blue", linestyle="--", #type: ignore
                    linewidth=2, label=f"95% CI Upper: {ci_upper:.2f}")

        plt.title(f"Bootstrap das Médias Amostrais - {name}", fontsize=14)
        plt.xlabel(f"Valor médio de {col_target} %", fontsize=12)
        plt.ylabel("Densidade de Probabilidade", fontsize=12)
        plt.grid(alpha=0.3)

        plt.legend(fontsize = 8.5, framealpha = 0.2, loc='upper right', title="Estatísticas")
        plt.tight_layout()

        plt.show()


    plt.figure(figsize=(14, 8))

    for name, data in stored_results.items():
        means = data['means']
        color = data['color']
        mean_est, ci_lower, ci_upper = data['stats']

        label_str = f"{name}\n(Média: {mean_est:.2f}, IC: [{ci_lower:.2f}, {ci_upper:.2f}])"

        sns.histplot(
            means,
            color = color,
            kde =True,
            element='step',
            stat='density',
            alpha=0.25,
            label=label_str
        )
        plt.axvline(mean_est, color = color, linestyle="--", linewidth=2, alpha = 0.8) #type: ignore

    plt.title("Comparação das Distribuições Bootstrap", fontsize=14)
    plt.xlabel(f"Valor médio de {col_target} %", fontsize=12)
    plt.ylabel("Densidade de Probabilidade", fontsize=12)
    plt.grid(alpha=0.3)

    plt.legend(fontsize = 8.5, framealpha = 0.2, loc='upper right', title="Estatísticas")
    plt.tight_layout()
    plt.show()


bootstrap_charts(
    df1, df1_name,
    df2, df2_name,
    df3, df3_name,
    "coverage_score",
    n_boot = 5000
)


# %% [markdown]
# ### 3. Separação entre mínimo local

# %%
def lms(df, dataset_name, col_score='coverage_score', max_search=25) -> float: #type: ignore
    """
    Function to find the local minimum between 0 and max_search and plot the density curve
    Returns the cutoff point
    """
    data = df[col_score].dropna()

    #(simula a curva do gráfico)
    kde = gaussian_kde(data)

    x_grid = np.linspace(2, max_search, 1000)
    y_density = kde(x_grid)

    idx_min = np.argmin(y_density)
    cutoff_point = x_grid[idx_min]

    plt.figure(figsize=(8, 4))
    plt.plot(x_grid, y_density, label='Densidade')
    plt.axvline(cutoff_point, color='red', linestyle='--', label=f'Vale: {cutoff_point:.2f}%')
    plt.title(f"Cutoff Point Identified in: {dataset_name} at {cutoff_point:.2f}%")
    plt.legend()
    plt.show()

    return cutoff_point

# Execução
cut1 = lms(df1, dataset_name=df1_name, col_score='coverage_score')
cut2 = lms(df2, dataset_name=df2_name, col_score='coverage_score')
cut3 = lms(df3, dataset_name=df3_name, col_score='coverage_score')

# %% [markdown]
# Os 3 gráficos acima mostram os pontos mínimos para cada um das curvas sendo estudadas,
# agora podemos separar entre antes desse ponto e depois desse ponto para avaliar os
# exemplos, e ambos estão armazenados em cut1 e cut2, que será usado para poder separarmos
# nosso ponto para as seguintes análises
#

# %%
df1_low_coverage = df1[df1['coverage_score'] <= cut1]
df1_high_coverage = df1[df1['coverage_score'] > cut1]

df2_low_coverage = df2[df2['coverage_score'] <= cut2]
df2_high_coverage = df2[df2['coverage_score'] > cut2]

df3_low_coverage = df3[df3['coverage_score'] <= cut3]
df3_high_coverage = df3[df3['coverage_score'] > cut3]


# %% [markdown]
# Agora que nossos df com os cortes estão definidos passarei por todas as análises
# que achei poderiam ter relevância pro nosso estudo.
# Comentários são agradecidos e bem-vindos, caso surgirem novas ideias
# também podem serem adicionadas.

# %% [markdown]
# ### 4. Análise de Head Words
#
# O objetivo da análise será verificar se existe alguma relevância entre a
#  forma que as queries foram feitas, por exemplo, verificar se queries
#  sendo mais concisas impacta diretamente na procura da URL.

# %%
def analyze_intent_head_words(df_low, df_high, dataset_name, top_n = 20) -> None: #type: ignore
    """
    Analyzes and plots Head Words between the low and high coverage datasets.
    """
    possible_cols = [col_query,prompt_query_fan_out]
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    def get_head_words_counts(df_target) -> pd.Series: #type: ignore
        for col in possible_cols:
            if col in df_target.columns:
                target_col = col
                break

        return df_target[target_col].astype(str).dropna().apply( #type: ignore
            lambda x: unidecode(x.lower().split()[0]) if len(x.split()) > 0 else None
        ).value_counts().head(top_n)

    counts_low = get_head_words_counts(df_low)

    if not counts_low.empty:
        sns.barplot(x=counts_low.values, y=counts_low.index, ax=axes[0], palette="viridis")
        axes[0].set_title(
            f"Top {
                top_n
                } Head Words - Low Coverage\n(Total: {len(df_low)}) in dataset {dataset_name}"
            )
        axes[0].set_xlabel("Frequency")
        axes[0].set_ylabel("Head Words")

        for i, v in enumerate(counts_low.values):
            axes[0].text(v + 0.5, i, str(v), color='black', va='center', fontweight='bold')

    else:
        axes[0].text(0.5, 0.5, 'No data available', ha = 'center')

    counts_high = get_head_words_counts(df_high)

    if not counts_high.empty:
        sns.barplot(x=counts_high.values, y=counts_high.index, ax=axes[1], palette="viridis")
        axes[1].set_title(
            f"Top {
                top_n
                } Head Words - High Coverage\n(Total: {len(df_high)}) in dataset {dataset_name}"
            )
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Head Words")

        for i, v in enumerate(counts_high.values):
            axes[1].text(v + 0.5, i, str(v), color='black', va='center', fontweight='bold')

    else:
        axes[1].text(0.5, 0.5, 'No data available', ha = 'center')

    plt.suptitle(f"Comparison of Head Words in Low and High Coverage - {dataset_name}")
    plt.tight_layout()
    plt.show()

analyze_intent_head_words(df1_low_coverage, df1_high_coverage, dataset_name= df1_name, top_n=20)
analyze_intent_head_words(df2_low_coverage, df2_high_coverage, dataset_name= df2_name, top_n=20)
analyze_intent_head_words(df3_low_coverage, df3_high_coverage, dataset_name= df3_name, top_n=20)


# %% [markdown]
# Com os resultados acima podemos ver que são muito similar e que,
# a pesar do corte os resultados foram muito similares,
# o que da a entender que o top head não teve muita importância no query.

# %% [markdown]
# ### 5. Dominios Predominantes
#
# Como seguinte passo agora iremos analisar os domínios.
# O objetivo é investigar se existem dominios especificos que predominam cada grupo

# %%
def analyze_domain_comparative(df_low, df_high, dataset_name, top_n = 10) -> None: #type: ignore
    """
    tbd
    """

    _, axes = plt.subplots(1, 2, figsize=(16, 7))

    def plot_top_sites(df_target, ax, title, color) -> None: #type: ignore
        top_sites = df_target['url_hostname'].value_counts().head(top_n)
        sns.barplot(x=top_sites.values, y = top_sites.index, ax = ax, palette=color)
        ax.set_title(f"Top {top_n} Domains - {title}\n({dataset_name})")
        ax.set_xlabel("Quantity")

        max_x = top_sites.values.max()
        ax.set_xlim(0, max_x * 1.25)

        for i, v in enumerate(top_sites.values):
            pct = (v/len(df_target)) * 100
            ax.text(
                v + max_x * 0.01, i,
                     f"{v} ({pct:.1f}%)", color='black', va='center', fontweight='bold'
                     )

    if not df_low.empty:
        plot_top_sites(
            df_low,
            axes[0],
            title=f"Low Coverage\n(Total = {len(df_low)})", color="Blues")

    if not df_high.empty:
        plot_top_sites(
            df_high,
            axes[1],
            title=f"High Coverage\n(Total: {len(df_high)})", color="Greens")

    plt.suptitle(f"Comparison of Top Domains in Low and High Coverage - {dataset_name}")
    plt.tight_layout()
    plt.show()

analyze_domain_comparative(df1_low_coverage, df1_high_coverage, dataset_name=df1_name, top_n=15)
analyze_domain_comparative(df2_low_coverage, df2_high_coverage, dataset_name=df2_name, top_n=15)
analyze_domain_comparative(df3_low_coverage, df3_high_coverage, dataset_name=df3_name, top_n=15)


# %% [markdown]
# Como podemos ver acima o grupo semántico (Low Coverage)
# é dominado pela Wikipedia, o que pode nos dar indicio que existe
#  uma maior importância ao conteúdo do site ou à simplicidade
#  da URL da wikipedia e da query;
#
# Para o grupo literal (high coverage) podemos ver alguns exemplos
# que dão a entender que podem ser sites de dependem de keyword
# matching, que também pode ser estudado.

# %% [markdown]
# ## 2. Os termos que aparecem no query, aparecem na URL ou no slug?

# %% [markdown]
# ### 1. Top slugs matching com query

# %% [markdown]
# Em base a esses resultados agora queremos testar quais foram
# aquelas palavras mais comuns que apareceram em query, url e slugs.
#  A seguir também iremos procurar quais foram aqueles sites que foram
#   procurados mais vezes pelo agente.

# %%
def common_words(df, dataset_name, col_common_words='common_words_url', top_n=25) -> None: #type: ignore
    """
    Plots the most common words found both in Query and URL (or URL_hostname)
    """

    all_words = []

    for words_list in df[col_common_words]:
        if isinstance(words_list, str):
            try:
                list_words = ast.literal_eval(words_list)
            except(ValueError, SyntaxError):
                list_words = []
        elif isinstance(words_list, list):
            list_words = words_list
        else:
            list_words = []

        all_words.extend(list_words)
    if not all_words:
        print(f"No common words found in {dataset_name}.")
        return

    counter = Counter(all_words)
    top_words = counter.most_common(top_n)
    words = [x[0] for x in top_words]
    counts = [x[1] for x in top_words]
    total_matches = len(all_words)

    plt.figure(figsize =(10, 6))
    barplot = sns.barplot(x=counts, y=words, palette="viridis", hue = words, legend = False)
    plt.title(f"Top {
        top_n
        } words that match in Query and URL - {dataset_name}\n(Total Matches: {total_matches})")
    plt.xlabel("Frequency")
    plt.ylabel("Words")

    max_x = counts[0]
    plt.xlim(0, max_x * 1.25)

    for i, v in enumerate(counts):
        pct = (v/total_matches * 100)
        label_text = f"{v} ({pct:.1f}%)"
        offset = max_x * 0.01

        barplot.text(v + offset, i, label_text, color='black', va='center', fontweight='bold')
    plt.show()

common_words(df1, df1_name, col_common_words='common_words_url', top_n=30)
common_words(df2, df2_name, col_common_words='common_words_url', top_n=30)
common_words(df3, df3_name, col_common_words='common_words_url', top_n=30)



# %% [markdown]
# No gráfico acime vemos as top 25 palavras mais comuns que bateram com query/URL.
#  Para verificar o peso que essas palavras têm no query seria
#  necessária uma análise de sentimento, que pode ser um passo a avaliar no futuro.

# %% [markdown]
# ### 2. Dominios Predominantes

# %% [markdown]
# A seguir, vamos a identificar se nosso agente tem vies por algum
#  site a procurar as informações solicitadas.

# %%
def top_url_names(df,dataset_name,top_n = 25) -> None: #type: ignore
    """
    Plots the most common URL hostnames in the dataset.
    """

    total_data = len(df)
    top_sites = df['url_hostname'].value_counts().head(top_n)
    plt.figure(figsize =(12, 7))
    barplot2 = sns.barplot(x=top_sites.values, y=top_sites.index, palette="mako")
    max_x = top_sites.values.max()

    plt.xlim(0, max_x * 1.25)
    plt.title(f"Top {top_n} URL Hostnames - {dataset_name}\n(Total Entries: {total_data})")
    plt.xlabel("URL Quantity")
    plt.ylabel("URL Hostnames")

    for i, v in enumerate(top_sites.values):
        pct = (v/total_data * 100)
        label_text = f"{v} ({pct:.1f}%)"
        offset = max_x * 0.01
        barplot2.text(v + offset, i, label_text, color='black', va='center', fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

top_url_names(df1, df1_name, top_n = 25)
top_url_names(df2, df2_name, top_n = 25)
top_url_names(df3, df3_name, top_n = 25)


# %% [markdown]
# ### 3. Teste - Crawler Type Bias

# %% [markdown]
# Nessa seção iremos criar umas análises iguais mas comparando
# o tipo de crawler para identificar se existe algum bias ao
# ser do UI ou da API

# %% [markdown]
# Nesta etapa, investigamos se a infraestrutura de coleta (o tipo de Crawler)
# influencia a qualidade das URLs recuperadas. A hipótese é verificar se a
# Interface de Chat (UI) e a API possuem comportamentos
# distintos de retrieval.
#
# Para isso, utilizamos a métrica anteriormente usada de Coverage Score,
# que mede a porcentagem de palavras-chave da pergunta original
# que aparecem explicitamente na string da URL retornada.
#
# Lembrando que para este caso, a query trata-se da query do agente,
# não do usuário.

# %% [markdown]
# O primeiro passo foi verificar a "agressividade"
# de cada crawler, ou seja, quantos links
#  cada um retorna por pergunta.

# %%
def analyze_crawler_fanout(df, dataset_name) -> None: #type: ignore
    fanout_data = df.groupby(
        ['crawler_type', 'prompt_hash']
        ).size().reset_index(name='citation_count')

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=fanout_data,
                x='crawler_type',
                y='citation_count',
                palette="Set2",
                showfliers=False)

    plt.title(f"Distribuição de Citações por Prompt (UI vs API) - {dataset_name}")
    plt.ylabel("Número de Citações por Resposta")
    plt.xlabel("Tipo de Crawler")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    print(fanout_data.groupby('crawler_type')['citation_count'].describe())

analyze_crawler_fanout(df3, df3_name)


# %% [markdown]
# O gráfico boxplot indica um comportamento determinístico
# : a API busca um número fixo e pequeno de fontes.
#
# Já o boxplot da UI mostra pontos atingindo mais de
#  8.000 citações para uma única resposta.

# %% [markdown]
# Por enquanto, a quantidade de dados de tipo UI é muito maior
#  à quantidade de dados de tipo API, pelo qual recomendaria
#   que o estudo seja feito novamente no futuro ou com um df
#   com dados mais similares ao outros, continuaremos nossa
#   análise para simplificar a implementação no futuro.

# %%
def analyze_domains_split_by_crawler(df, top_n=10) -> None: #type: ignore
    """
    Generates two side-by-side bar charts comparing top domains for API vs UI crawlers.
    """
    _, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)

    crawler_types = ['api', 'ui']
    palettes = ['mako', 'rocket']

    for i, c_type in enumerate(crawler_types):
        ax = axes[i]

        df_subset = df[df['crawler_type'] == c_type]

        n_total = len(df_subset)

        top_counts = df_subset['url_hostname'].value_counts().head(top_n)

        if top_counts.empty:
            ax.text(0.5, 0.5, f"Sem dados para {c_type}", ha='center')
            continue

        sns.barplot(x=top_counts.values, y=top_counts.index, ax=ax, palette=palettes[i])

        ax.set_title(
            f"Top {
                top_n
                } Domínios - {
                    c_type.upper()
                    }\n(Total de Citações: {n_total})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Quantidade de Citações")
        ax.set_ylabel("Domínio")

        max_x = top_counts.values.max()
        ax.set_xlim(0, max_x * 1.25)

        for index, value in enumerate(top_counts.values):
            pct = (value / n_total) * 100
            label = f"{value} ({pct:.1f}%)"

            ax.text(
                value + (max_x * 0.02),
                index, label, va='center',
                fontweight='bold', color='#333333')

        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

analyze_domains_split_by_crawler(df3, top_n=15)


# %% [markdown]
# *MUDARÁ COM O TEMPO MAS CONCLUSÕES ATUAIS SÃO AS SEGUINTES*:
#
# API prefere fontes primarias e sites com nichos especificos.
#
# UI é dominada por sites que listam muitas empresas ou dados.

# %%
def compare_crawler_bias(df, dataset_name) -> None: #type: ignore
    """
    Compare coverage scores between UI and API crawlers.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='crawler_type', y='coverage_score', palette="Set2", showfliers=False)
    plt.title(f"Distribuição de Cobertura: UI vs API\n({dataset_name})")
    plt.ylabel("Score de Cobertura (% de Match)")
    plt.xlabel("Tipo de Crawler")
    plt.grid(axis='y', alpha=0.3)

    medias = df.groupby('crawler_type')['coverage_score'].mean()
    for i, (_, valor) in enumerate(medias.items()):
        plt.text(i, valor + 2,
                  f"Média: {valor:.1f}%",
                    ha='center', fontweight='bold', color='black')

    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df,
                x='coverage_score',
                  hue='crawler_type',
                    fill=True, common_norm=False,
                      palette="Set2", alpha=0.4)
    plt.title(f"Score Density\n({dataset_name})")
    plt.xlabel("Score de Cobertura (0-100)")
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()

compare_crawler_bias(df3, df3_name)

# %% [markdown]
# Boxplot (Nível de Especificidade)
# UI (Laranja): Apresenta uma média superior (8.3%).
#  A caixa (box) é mais alta e alongada, indicando que a UI varia mais,
#   mas frequentemente atinge scores de 10% a 15%. Isso sugere um esforço
#    maior em buscar a página exata do produto ou notícia.
#
# API (Verde): Média inferior (6.2%). A caixa é comprimida na parte
#  inferior, indicando que a grande maioria das respostas
#   tem baixa correspondência textual na URL.

# %% [markdown]
# Densidade
#
# O "Pico" da API (Verde):
# A montanha alta colada no 0 confirma que a API
# prioriza Homepages e Domínios Raiz (ex: www.gov.br).
# Ela entrega a fonte correta, mas não necessariamente a página específica.
#
# As "Ondas" da UI (Laranja):
#  A curva laranja tem picos secundários (ondulações)
#  entre os scores 5 e 20. Isso confirma tecnicamente
#  que a UI realiza Deep Linking. Ela tende a "mergulhar"
#   no site para trazer URLs longas e descritivas.
