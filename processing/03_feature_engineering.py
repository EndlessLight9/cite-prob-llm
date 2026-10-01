import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS
# =====================================================================
path = Path(__file__).parent.parent

path_auditoria = path / "data" / "gerador_vetores.csv" 
pasta_processed = path / "data" / "processed"

# path to save the final vectors
try:
    pasta_processed.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Erro ao criar a pasta de saída: {e}")
    exit()

pasta_processed = path / "data" / "processed"
pasta_processed.mkdir(parents=True, exist_ok=True)

print("Lendo auditoria e iniciando Feature Engineering (V1 a V5)...")

# Carrega a tabela de auditoria
try:
    df_auditoria = pd.read_csv(path_auditoria)
except FileNotFoundError:
    print(f"Erro: O ficheiro {path_auditoria} não foi encontrado. Rode o script 02 primeiro.")
    exit()

# =====================================================================
# 2. A MAGIA MATEMÁTICA: CÁLCULO DAS VARIÁVEIS (V1 A V5)
# =====================================================================

# V1: Taxa de Presença
df_auditoria['V1_taxa_presenca'] = df_auditoria['url_apareceu_n_iteracoes'] / df_auditoria['total_iteracoes_cenario']

# V2: Dominância (Market Share)
df_auditoria['V2_market_share'] = np.where(
    df_auditoria['total_citacoes_cenario'] > 0,
    df_auditoria['url_total_citacoes'] / df_auditoria['total_citacoes_cenario'],
    0.0
)

# V3A: Posição Média interessante
df_auditoria['V3A_position_important'] = df_auditoria['url_posicao_media_bruta'].fillna(15.0)

""" # V3B: Posição Média ignorando nulos (para comparação)
mean_position = df_auditoria['url_posicao_media_bruta'].mean()
df_auditoria['V3B_position_ignored'] = df_auditoria['url_posicao_media_bruta'].fillna(mean_position)
"""
# V3B: Posição Média ignorando nulos (para comparação)
# Se apareceu apenas 1 vez, tratamos a posição como não confiável e ignoramos.
posicao_para_v3b = df_auditoria['url_posicao_media_bruta'].where(
    df_auditoria['url_total_citacoes'] > 2,
    np.nan
)

# Mantém NaN para posições não confiáveis e normaliza por cenário
# (source + prompt_limpo), invertendo a polaridade para 0..1.
# Regra: posição 1 = melhor (score 1), pior posição observada no cenário = score 0.
group_cols = ['source', 'prompt_limpo']
pior_posicao_real = posicao_para_v3b.groupby([df_auditoria[c] for c in group_cols]).transform('max')
melhor_posicao_fixa = 1.0
amplitude = pior_posicao_real - melhor_posicao_fixa

v3b_normalizado = pd.Series(np.nan, index=df_auditoria.index, dtype='float64')
mascara_valida = posicao_para_v3b.notna()
mascara_sem_variacao = mascara_valida & amplitude.eq(0)
mascara_com_variacao = mascara_valida & amplitude.gt(0)

# Se a pior posição observada também for 1, todo mundo ficou no topo.
v3b_normalizado.loc[mascara_sem_variacao] = 1.0
v3b_normalizado.loc[mascara_com_variacao] = (
    (pior_posicao_real.loc[mascara_com_variacao] - posicao_para_v3b.loc[mascara_com_variacao])
    / amplitude.loc[mascara_com_variacao]
)

df_auditoria['V3B_position_ignored'] = v3b_normalizado

# V4: Taxa de Conversão Frontend
df_auditoria['V4_taxa_conversao'] = np.where(
    df_auditoria['url_apareceu_n_iteracoes'] > 0,
    df_auditoria['url_total_virou_link'] / df_auditoria['url_apareceu_n_iteracoes'],
    0.0
)
# Garante que não passa de 100% caso a IA gere links duplicados da mesma marca
df_auditoria['V4_taxa_conversao'] = df_auditoria['V4_taxa_conversao'].clip(upper=1.0)

# V5: Estabilidade (Tratamento de Desvio Nulo)
df_auditoria['V5_estabilidade_desvio'] = df_auditoria['url_desvio_padrao_posicao'].fillna(0.0)

# =====================================================================
# V5 - COLUNA AUXILIAR: CLASSIFICAÇÃO DO TIPO DE ZERO
# =====================================================================
# Existem 3 tipos de V5=0 que precisam ser distinguidos:
#
#  CASO 1 - ZERO REAL (estável de verdade):
#    URL apareceu múltiplas vezes E sempre na mesma posição
#    Exemplo: apareceu 50x, sempre na posição 3 → std = 0.0
#    Condição: url_total_citacoes >= 5 AND url_desvio_padrao_posicao == 0.0
#
#  CASO 2 - ZERO ARTIFICIAL (dados insuficientes):
#    URL apareceu poucas vezes → não há dados para calcular variação
#    Exemplo: apareceu 1x na posição 5 → std = NaN → 0.0
#    Condição: url_total_citacoes < 5 AND url_desvio_padrao_posicao era NaN
#
#  CASO 3 - ZERO ARTIFICIAL (sem posição alguma):
#    URL nunca teve link com posição (sem links_attached com position)
#    Exemplo: apareceu na citations mas nunca virou link visível → std = NaN → 0.0
#    Condição: url_total_virou_link == 0

LIMIAR_MINIMO_CITACOES = 5  # Mínimo de citações para considerar V5 confiável

def classificar_v5(row):
    sem_posicao = row['url_total_virou_link'] == 0
    poucos_dados = row['url_total_citacoes'] < LIMIAR_MINIMO_CITACOES
    desvio_nulo = pd.isna(row['url_desvio_padrao_posicao']) or row['url_desvio_padrao_posicao'] == 0.0

    if sem_posicao:
        return 'CASO_3_sem_posicao'
    elif poucos_dados:
        return 'CASO_2_dados_insuficientes'
    elif desvio_nulo:
        return 'CASO_1_zero_real'
    else:
        return 'V5_valido'

df_auditoria['V5_tipo'] = df_auditoria.apply(classificar_v5, axis=1)

# V5 confiável: usa o valor real apenas para CASO_1 e V5_valido
# Para os demais, preenche com NaN para não contaminar o K-Means
df_auditoria['V5_confiavel'] = np.where(
    df_auditoria['V5_tipo'].isin(['CASO_1_zero_real', 'V5_valido']),
    df_auditoria['V5_estabilidade_desvio'],
    np.nan
)


# =====================================================================
# 3. EXPORTAÇÃO DOS VETORES (PREPARAÇÃO PARA O K-MEANS)
# =====================================================================
print("Fatiando os cenários e salvando os vetores...")

# Mantemos apenas as colunas de "DNA"
colunas_dna = [
    'source', 'prompt_limpo', 'url_hostname',
    'V1_taxa_presenca', 'V2_market_share', 'V3A_position_important', 'V3B_position_ignored',
    'V4_taxa_conversao', 'V5_estabilidade_desvio', 'V5_tipo', 'V5_confiavel'
]
df_vetores = df_auditoria[colunas_dna].copy()

# Separamos um ficheiro CSV por Cenário (IA + Prompt) para o ML não misturar os mercados
cenarios = df_vetores[['source', 'prompt_limpo']].drop_duplicates()

for _, cenario in cenarios.iterrows():
    ia = cenario['source']
    prompt = cenario['prompt_limpo']
    
    # Filtra apenas os dados deste cenário
    df_cenario = df_vetores[(df_vetores['source'] == ia) & (df_vetores['prompt_limpo'] == prompt)]
    
    # Remove as colunas de contexto, deixando apenas a URL e os 5 Vs, ordenado por Presença
    df_limpo = df_cenario.drop(columns=['source', 'prompt_limpo']).sort_values('V1_taxa_presenca', ascending=False)
    
    # Cria um nome de ficheiro seguro e salva
    prompt_curto = "".join([c for c in prompt[:30] if c.isalnum() or c==' ']).strip()
    nome_ficheiro = f"vetor_FINAL_{ia}_{prompt_curto}.csv"
    caminho_saida = pasta_processed / nome_ficheiro
    
    df_limpo.to_csv(caminho_saida, index=False)
    print(f"  -> Salvo: {nome_ficheiro}")

print("\nFeature Engineering Concluída!")