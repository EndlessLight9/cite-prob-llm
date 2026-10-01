import pandas as pd
import numpy as np
import ast
import re

from pathlib import Path

path = Path(__file__).parent.parent
path_master = path / "data" / "master_dataset_tcc.csv"
df_master = pd.read_csv(path_master)

# =====================================================================

# =====================================================================
# 1. CARREGAMENTO E LIMPEZA INICIAL
# =====================================================================
# df_master = pd.read_csv("caminho_do_seu_dataset_master.csv")

print("Iniciando a extração dos números brutos...")

# Remove as "linhas fantasmas" (onde a IA não citou a marca)
#df_master = df_master[df_master['frequencia_citacao'] > 0].copy()

# Limpa o Cache-Busting do prompt
df_master['prompt_limpo'] = df_master['prompt_query'].apply(
    lambda x: re.sub(r'\[ignorar_id:.*?\]', '', str(x)).strip()
)

# Extrai a melhor posição (apenas para termos a média bruta)
def extrair_melhor_posicao(pos_string) -> float:
    try:
        lista_real = ast.literal_eval(str(pos_string))
        if isinstance(lista_real, list) and len(lista_real) > 0:
            return min(lista_real)
        return np.nan
    except (ValueError, SyntaxError):
        return np.nan

df_master['melhor_posicao'] = df_master['posicoes_links'].apply(extrair_melhor_posicao)


# =====================================================================
# 2. CÁLCULO DOS DENOMINADORES TOTAIS (POR CENÁRIO)
# =====================================================================
# Calcula os totais por cenário para servir de referência na análise final.
totais_cenario = df_master.groupby(['source', 'prompt_limpo']).agg(
    total_iteracoes_cenario=('test_id', 'nunique'),
    total_citacoes_cenario=('frequencia_citacao', 'sum')
).reset_index()


# =====================================================================
# 3. CÁLCULO DOS NUMERADORES (POR URL)
# =====================================================================
# Quantas iterações únicas a URL apareceu? (Para validar V1)
presenca_unica = df_master.drop_duplicates(subset=['source', 'prompt_limpo', 'test_id', 'url_hostname'])
contagem_presenca = presenca_unica.groupby(['source', 'prompt_limpo', 'url_hostname']).size().reset_index(name='url_apareceu_n_iteracoes')

# Quantas citações totais, conversões e posições a URL teve? (Para validar V2, V3, V4)
metricas_brutas_url = df_master.groupby(['source', 'prompt_limpo', 'url_hostname']).agg(
    url_total_citacoes=('frequencia_citacao', 'sum'),
    url_total_virou_link=('link_mentioned_in_response', 'sum'),
    url_posicao_media_bruta=('melhor_posicao', 'mean'),
    url_desvio_padrao_posicao=('melhor_posicao', 'std')
).reset_index()


# =====================================================================
# 4. CONSOLIDAÇÃO DA TABELA DE AUDITORIA
# =====================================================================
# Juntamos as métricas da URL
df_auditoria = pd.merge(contagem_presenca, metricas_brutas_url, on=['source', 'prompt_limpo', 'url_hostname'])

# Junta os totais do cenário para consolidar a visão geral do cenário.
df_auditoria = pd.merge(df_auditoria, totais_cenario, on=['source', 'prompt_limpo'])

# Reorganizando as colunas para a leitura ficar lógica e fluida no Excel/VS Code
colunas_finais = [
    'source', 
    'prompt_limpo', 
    'url_hostname', 
    'url_apareceu_n_iteracoes', 
    'total_iteracoes_cenario',
    'url_total_citacoes',
    'total_citacoes_cenario',
    'url_total_virou_link',
    'url_posicao_media_bruta',
    'url_desvio_padrao_posicao'
]
df_auditoria = df_auditoria[colunas_finais]

# Ordenar por IA, Prompt e quem apareceu mais vezes (para os principais ficarem no topo)
df_auditoria = df_auditoria.sort_values(by=['source', 'prompt_limpo', 'url_apareceu_n_iteracoes'], ascending=[True, True, False])

# Salvar o arquivo
name_arquivo = "gerador_vetores.csv"
# guardar na pasta data
pasta_dados = path / "data"
if not pasta_dados.exists():
    pasta_dados.mkdir()
name_arquivo = pasta_dados / name_arquivo
df_auditoria.to_csv(name_arquivo, index=False)
print(f"Arquivo '{name_arquivo}' gerado com sucesso!")