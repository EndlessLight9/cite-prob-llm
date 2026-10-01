import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================================================
# 1. CONFIGURAÇÃO
# =====================================================================
path = Path(__file__).parent.parent
pasta_processed = path / "data" / "processed"
pasta_resultados = path / "data" / "results"
pasta_graficos = pasta_resultados / "graficos" / "Histogramas"
pasta_graficos.mkdir(parents=True, exist_ok=True)

print("Gerando histogramas para todos os atributos (V1-V5)...\n")

# Nomes dos atributos
atributos = {
    'V1_taxa_presenca': 'V1: Taxa de Presença',
    'V2_market_share': 'V2: Market Share',
    'V3B_position_ignored': 'V3: Posição Média',
    'V4_taxa_conversao': 'V4: Taxa Conversão',
    'V5_confiavel': 'V5: Estabilidade'
}

def get_prompt_nome(nome_arquivo):
    """Mapeia nome do arquivo para prompt legível."""
    if 'empresas de' in nome_arquivo:
        return 'Prompt1_GEO'
    elif 'marcas d' in nome_arquivo:
        return 'Prompt2_Fertilizante'
    elif 'operador' in nome_arquivo:
        return 'Prompt3_Operadoras'
    else:
        return 'PromptUnknown'

# =====================================================================
# 2. FUNÇÃO: CRIAR HISTOGRAMAS PARA UM CENÁRIO
# =====================================================================
def criar_histogramas(df, ia, prompt_nome):
    """
    Cria 5 histogramas (um para cada atributo V1-V5).
    Mostra a distribuição geral e por cluster.
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{ia} - {prompt_nome} | Distribuição dos Atributos',
                 fontsize=16, fontweight='bold', y=0.995)

    axes = axes.flatten()

    atributos_uso = atributos.copy()
    if 'V5_confiavel' not in df.columns and 'V5_estabilidade_desvio' in df.columns:
        atributos_uso['V5_estabilidade_desvio'] = 'V5: Estabilidade (Bruta - Fallback)'

    for idx, (col, titulo) in enumerate(atributos_uso.items()):
        ax = axes[idx]
        dados_coluna = df[col].dropna()

        # Se a coluna estiver toda vazia (NaN), evita erro de range no matplotlib.
        if dados_coluna.empty:
            ax.text(0.5, 0.5, 'Sem dados validos', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
            ax.set_xlabel(titulo, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequencia', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            stats_text = f'Media: nan\nMediana: nan\nDesvio: nan\nN validos: 0\nN total: {len(df)}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            continue

        # Histograma geral com overlay de densidade (com tratamento de erro)
        ax.hist(dados_coluna, bins=30, color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Adicionar linha de densidade (com try-except para dados singulares)
        try:
            from scipy import stats
            density = stats.gaussian_kde(dados_coluna)
            xs = np.linspace(dados_coluna.min(), dados_coluna.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(xs, density(xs), 'r-', linewidth=2, label='Densidade')
            ax2.set_ylabel('Densidade', fontsize=10, color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # Expande o teto do eixo Y secundário (Densidade) para não sobrepor a caixa de texto
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_ylim(ymin2, ymax2 * 1.35)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            # Se falhar (dados singulares), apenas usar o histograma
            pass

        # Configurar eixos
        ax.set_xlabel(titulo, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequência', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Expande o teto do eixo Y principal (Histograma) em 35% para acomodar as estatísticas
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.35)

        # Adicionar estatísticas
        media = dados_coluna.mean()
        mediana = dados_coluna.median()
        desvio = dados_coluna.std()

        stats_text = (
            f'Media: {media:.3f}\nMediana: {mediana:.3f}\nDesvio: {desvio:.3f}'
            f'\nN validos: {len(dados_coluna)}\nN total: {len(df)}'
        )
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remover o 6º subplot (deixar em branco)
    fig.delaxes(axes[5])

    plt.tight_layout()

    # Salvar figura
    nome_arquivo = f"{ia}_{prompt_nome}_Histogramas.png"
    caminho_saida = pasta_graficos / nome_arquivo
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_saida

# =====================================================================
# 3. FUNÇÃO: CRIAR HISTOGRAMAS COM SEPARAÇÃO POR CLUSTER
# =====================================================================
def criar_histogramas_por_cluster(df_original, df_clusterizado, ia, prompt_nome):
    """
    Cria histogramas separados por cluster para visualizar como cada
    cluster se diferencia em cada atributo.
    """

    df = df_original.copy()
    df['Cluster'] = df_clusterizado['Cluster_ID'].values

    n_clusters = df['Cluster'].nunique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{ia} - {prompt_nome} | Histogramas por Cluster',
                 fontsize=16, fontweight='bold', y=0.995)

    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    atributos_uso = atributos.copy()
    if 'V5_confiavel' not in df.columns and 'V5_estabilidade_desvio' in df.columns:
        atributos_uso['V5_estabilidade_desvio'] = 'V5: Estabilidade (Bruta - Fallback)'

    for idx, (col, titulo) in enumerate(atributos_uso.items()):
        ax = axes[idx]
        houve_dados = False

        # Plotar histograma para cada cluster
        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster_id][col].dropna()
            if cluster_data.empty:
                continue
            houve_dados = True
            ax.hist(cluster_data, bins=20, alpha=0.5, label=f'Cluster {cluster_id}',
                   color=colors[cluster_id], edgecolor='black', linewidth=0.5)

        ax.set_xlabel(titulo, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequência', fontsize=11, fontweight='bold')
        
        # Expande o teto do eixo Y principal para abrir espaço para a legenda
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.35)
        
        if houve_dados:
            ax.legend(loc='upper right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Sem dados validos', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='gray')
        ax.grid(True, alpha=0.3, axis='y')

    # Remover o 6º subplot
    fig.delaxes(axes[5])

    plt.tight_layout()

    # Salvar figura
    nome_arquivo = f"{ia}_{prompt_nome}_Histogramas_PorCluster.png"
    caminho_saida = pasta_graficos / nome_arquivo
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_saida

# =====================================================================
# 4. PROCESSAR TODOS OS CENÁRIOS
# =====================================================================
arquivos_vetores = sorted((pasta_processed).glob("vetor_FINAL_*.csv"))
contador = 0
resumo = []

for arquivo_vetor in arquivos_vetores:
    contador += 1

    # Extrair IA e prompt
    nome = arquivo_vetor.name
    ia = 'GOOGLE_AI' if 'GOOGLE_AI' in nome else ('OPENAI' if 'OPENAI' in nome else 'PERPLEXITY')
    prompt_nome = get_prompt_nome(nome)

    print(f"[{contador}/9] {ia} - {prompt_nome}...", end=" ")

    # Carregar dados
    df_original = pd.read_csv(arquivo_vetor)

    # Encontrar arquivo clusterizado
    nome_cluster = nome.replace('vetor_FINAL_', 'Clusterizado_V3B_')
    arquivo_cluster = pasta_resultados / "V3B_Neutro" / nome_cluster

    if arquivo_cluster.exists():
        df_clusterizado = pd.read_csv(arquivo_cluster)

        # Gerar histogramas gerais
        caminho1 = criar_histogramas(df_original, ia, prompt_nome)

        # Gerar histogramas por cluster
        caminho2 = criar_histogramas_por_cluster(df_original, df_clusterizado, ia, prompt_nome)

        print("OK")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Status': 'OK'})
    else:
        print("Arquivo clusterizado não encontrado")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Status': 'ERRO'})

# =====================================================================
# 5. RESUMO FINAL
# =====================================================================
print(f"\n{'='*70}")
print("HISTOGRAMAS GERADOS COM SUCESSO")
print(f"{'='*70}\n")

df_resumo = pd.DataFrame(resumo)
print(df_resumo.to_string(index=False))

print(f"\n{'='*70}")
print("Localização: data/results/graficos/")
print(f"Total de histogramas: {len([f for f in (pasta_graficos).glob('*Histogramas*.png')])}")
print(f"{'='*70}\n")

print("Arquivos gerados:")
print("   - *_Histogramas.png (distribuição geral com densidade)")
print("   - *_Histogramas_PorCluster.png (distribuição separada por cluster)\n")