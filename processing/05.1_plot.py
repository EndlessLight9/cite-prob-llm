import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURACAO DE DIRETORIOS
# =====================================================================
path = Path(__file__).parent.parent

# LER OS DADOS QUE JA POSSUEM O CLUSTER_ID
pasta_clusterizados = path / "data" / "results" / "V3B_Neutro"

# Nova pasta para salvar as tabelas de medias
pasta_perfil = path / "data" / "results" / "tabelas"
pasta_perfil.mkdir(parents=True, exist_ok=True)

print("Gerando Tabelas de Perfil Medio por Cluster...\n")

def get_prompt_nome(nome_arquivo):
    """Mapeia nome do arquivo para prompt legivel."""
    if 'empresas de' in nome_arquivo:
        return 'Prompt1_GEO'
    elif 'marcas d' in nome_arquivo:
        return 'Prompt2_Fertilizante'
    elif 'operador' in nome_arquivo:
        return 'Prompt3_Operadoras'
    else:
        return 'PromptUnknown'

def exportar_tabela_imagem(df, titulo, caminho_saida):
    """
    Desenha o DataFrame como uma imagem de tabela usando Matplotlib
    e exporta em alta resolucao (.png).
    """
    # Ajusta o tamanho da figura dinamicamente para evitar sobreposicao entre colunas
    largura = max(14, len(df.columns) * 2.4)
    altura = len(df) * 0.6 + 1.8
    fig, ax = plt.subplots(figsize=(largura, altura))
    ax.axis('off')
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)

    # Formatar os numeros para exibicao limpa na imagem
    # Forma robusta e correta do Pandas para filtrar colunas float
    df_str = df.copy()
    colunas_float = df_str.select_dtypes(include=['float64', 'float32']).columns
    
    for col in colunas_float:
        # Tratamento robusto: formata apenas se nao for NaN
        df_str[col] = df_str[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")

    # Criar a tabela
    tabela = ax.table(cellText=df_str.values,
                      colLabels=df_str.columns,
                      loc='center',
                      cellLoc='center')
    tabela.auto_set_column_width(col=list(range(len(df_str.columns))))

    # Estilizar a tabela
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.35, 1.9) # Escala extra para aumentar espacamento horizontal/vertical

    # Colorir cabecalho e linhas alternadas
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1f497d') # Azul escuro para cabecalho
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white') # Linhas zebradas

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# =====================================================================
# 2. PROCESSAR TODOS OS CENARIOS CLUSTERIZADOS
# =====================================================================
arquivos_clusterizados = sorted(pasta_clusterizados.glob("*.csv"))

if not arquivos_clusterizados:
    print(f"Nenhum arquivo encontrado em: {pasta_clusterizados}")
    print("Certifique-se de ter rodado o script de K-Means (04) primeiro.")
    exit()

contador = 0
resumo = []

for arquivo in arquivos_clusterizados:
    contador += 1
    
    # Extrair IA e prompt
    nome = arquivo.name
    ia = 'GOOGLE_AI' if 'GOOGLE_AI' in nome else ('OPENAI' if 'OPENAI' in nome else 'PERPLEXITY')
    prompt_nome = get_prompt_nome(nome)
    
    print(f"[{contador}/{len(arquivos_clusterizados)}] Analisando {ia} - {prompt_nome}...", end=" ")
    
    # Carregar dados ja clusterizados
    df = pd.read_csv(arquivo)
    
    if 'Cluster_ID' not in df.columns:
        print("(Coluna Cluster_ID nao encontrada)")
        continue
        
    # Identificar automaticamente as colunas de variaveis (V1 a V5)
    # Ignoramos a coluna categorica 'V5_tipo' e garantimos o uso exclusivo do V3B
    colunas_v = [col for col in df.columns if col.startswith(('V1', 'V2', 'V3B', 'V4', 'V5')) and col != 'V5_tipo']
    
    # Evitar a duplicacao de colunas do V5
    if 'V5_confiavel' in colunas_v and 'V5_estabilidade_desvio' in colunas_v:
        colunas_v.remove('V5_estabilidade_desvio')
    
    # ---------------------------------------------------------
    # Agrupar por Cluster e calcular a Media
    # ---------------------------------------------------------
    # 1. Calcula a media das variaveis
    df_medias = df.groupby('Cluster_ID')[colunas_v].mean().reset_index()
    
    # 2. Conta quantas URLs cairam em cada cluster
    df_contagem = df.groupby('Cluster_ID').size().reset_index(name='Qtd_URLs')

    # 3. Junta tudo numa tabela so
    df_perfil = pd.merge(df_contagem, df_medias, on='Cluster_ID')
    
    # Arredondar para 4 casas decimais para calculo
    df_perfil = df_perfil.round(4)
    
    # Renomear as colunas
    renomeacoes = {
        'Cluster_ID': 'Cluster',
        'V1_taxa_presenca': 'V1_Media_Presenca',
        'V2_market_share': 'V2_Media_MarketShare',
        'V3B_position_ignored': 'V3_Media_ScorePosicao',
        'V4_taxa_conversao': 'V4_Media_Conversao',
        'V5_confiavel': 'V5_Media_Estabilidade',
        'V5_estabilidade_desvio': 'V5_Media_Estabilidade_Bruta' # Caso fallback seja usado
    }
    df_perfil = df_perfil.rename(columns=renomeacoes)
    
    # Ordenar pelos clusters (0, 1, 2, 3...)
    if 'Cluster' in df_perfil.columns:
        df_perfil = df_perfil.sort_values('Cluster', ascending=True)
    
    # ---------------------------------------------------------
    # Salvar outputs (CSV e PNG)
    # ---------------------------------------------------------
    # 1. Salvar a tabela em CSV
    nome_arquivo_base = f"{ia}_{prompt_nome}_Perfil_Clusters"
    caminho_csv = pasta_perfil / (nome_arquivo_base + ".csv")
    df_perfil.to_csv(caminho_csv, index=False, sep=';', decimal=',')
    
    # 2. Salvar a tabela em PNG
    titulo_tabela = f"Perfil Medio dos Clusters: {ia} | {prompt_nome}"
    caminho_png = pasta_perfil / (nome_arquivo_base + ".png")
    exportar_tabela_imagem(df_perfil, titulo_tabela, caminho_png)
    
    print("Salvo (CSV e PNG)!")
    resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Clusters': len(df_perfil)})

# =====================================================================
# 3. RESUMO FINAL
# =====================================================================
print(f"\n{'='*70}")
print("TABELAS DE PERFIL DE CLUSTERS GERADAS COM SUCESSO")
print(f"{'='*70}\n")

df_resumo = pd.DataFrame(resumo)
print(df_resumo.to_string(index=False))

print(f"\n{'='*70}")
print("Localizacao: data/results/tabelas/")
print("Arquivos gerados: .csv (para Excel) e .png (para o Word/LaTeX)")
print(f"{'='*70}\n")