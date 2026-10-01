import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================================================
# 1. CONFIGURAÇÃO
# =====================================================================
path = Path(__file__).parent.parent
pasta_processed  = path / "data" / "processed"
pasta_resultados = path / "data" / "results" / "V3B_Neutro"
pasta_graficos   = path / "data" / "results" / "graficos" / "tSNE"
pasta_graficos.mkdir(exist_ok=True)

FEATURES = ['V1_taxa_presenca', 'V2_market_share', 'V3B_position_ignored',
            'V4_taxa_conversao', 'V5_confiavel']

# Hiperparâmetros do t-SNE
# perplexity: balança atenção entre vizinhos próximos vs distantes
#   - valor baixo (5-15): foco em estruturas locais muito pequenas
#   - valor alto (30-50): foco em estruturas globais maiores
#   - regra geral: usar sqrt(n_amostras)
# n_iter: mais iterações = resultado mais estável, porém mais lento
PERPLEXITY_PADRAO = 30
N_ITER            = 2000

def get_prompt_nome(nome):
    if 'empresas de' in nome:
        return 'Prompt1_Empresas'
    if 'marcas d' in nome:
        return 'Prompt2_Marcas'
    if 'operador' in nome:
        return 'Prompt3_Operadoras'
    return 'PromptUnknown'


def alinhar_clusters_com_original(df_original, df_clusterizado):
    """
    Alinha Cluster_ID ao df_original usando uma chave estável de URL.
    Fallback para ordem de linhas apenas quando não houver chave única válida.
    """
    chaves_candidatas = ['url_hostname', 'url', 'url_canonica', 'url_link']

    for chave in chaves_candidatas:
        if chave in df_original.columns and chave in df_clusterizado.columns:
            if df_original[chave].is_unique and df_clusterizado[chave].is_unique:
                mapa = df_clusterizado.set_index(chave)['Cluster_ID']
                clusters = df_original[chave].map(mapa)
                if clusters.notna().all():
                    return clusters.astype(int).values, f'chave={chave}'

    if len(df_original) == len(df_clusterizado):
        return df_clusterizado['Cluster_ID'].values, 'fallback=ordem_linhas'

    raise ValueError(
        'Nao foi possivel alinhar Cluster_ID com df_original: sem chave unica e tamanhos diferentes.'
    )

# =====================================================================
# 2. FUNÇÃO: VISUALIZAR K-MEANS VIA t-SNE
# =====================================================================
def visualizar_tsne(df_original, df_clusterizado, ia, prompt_nome):
    """
    Reduz as 5 dimensões (V1-V5) para 2D via t-SNE e plota os clusters.

    t-SNE (t-Distributed Stochastic Neighbor Embedding):
    - Técnica não-linear: consegue capturar estruturas curvas e complexas
    - Preserva vizinhanças locais: pontos próximos no 5D ficam próximos no 2D
    - Diferente do PCA: não preserva distâncias globais — apenas grupos locais
    - Melhor para visualização, não para interpretação de distâncias absolutas
    """

    X        = df_original[FEATURES].copy()
    clusters, fonte_alinhamento = alinhar_clusters_com_original(df_original, df_clusterizado)
    k        = len(np.unique(clusters))
    n_urls   = len(X)

    # Imputar NaN e normalizar (mesmo processo do K-Means)
    imputer  = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Ajustar perplexity para datasets pequenos
    # t-SNE exige perplexity < n_amostras
    perplexity = min(30, n_urls / 3) #max(5, min(int(np.sqrt(n_urls)), 30)) 

    # --- APLICAR t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=N_ITER,
        random_state=42,
        learning_rate='auto',
        init='pca'
    )
    X_2d = tsne.fit_transform(X_scaled)

    # --- CRIAR FIGURA (2 subgráficos) ---
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    fig.suptitle(
        f'{ia} — {prompt_nome} | K-Means via t-SNE\n'
        f'(perplexity={perplexity}, n_iter={N_ITER}, {n_urls} URLs, {k} clusters | {fonte_alinhamento})',
        fontsize=14, fontweight='bold'
    )

    colors = plt.cm.tab10(np.arange(k) / 10)

    # ====== GRÁFICO 1: SCATTER t-SNE colorido por Cluster ======
    ax1 = axes[0]
    for cor_idx, cluster_id in enumerate(sorted(np.unique(clusters))):
        mask = clusters == cluster_id
        ax1.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[cor_idx]],
            label=f'Cluster {cluster_id} ({mask.sum()} URLs)',
            s=60, alpha=0.7, edgecolors='black', linewidth=0.3
        )

    ax1.set_xlabel('t-SNE Dimensão 1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimensão 2', fontsize=11, fontweight='bold')
    ax1.set_title('Clusters no espaço t-SNE', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    """     # Destacar top 5 URLs com label
    top5_idx = df_original['V1_taxa_presenca'].nlargest(5).index
    for idx in top5_idx:
        pos = np.where(df_original.index == idx)[0]
        if len(pos) > 0:
            i = pos[0]
            ax1.annotate(
                df_original.loc[idx, 'url_hostname'],
                (X_2d[i, 0], X_2d[i, 1]),
                fontsize=7, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7)
            ) """

    # ====== GRÁFICO 2: SCATTER t-SNE colorido por V1 (intensidade) ======
    ax2 = axes[1]
    sc = ax2.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=df_original['V1_taxa_presenca'],
        cmap='YlOrRd',
        s=60, alpha=0.8, edgecolors='black', linewidth=0.3
    )
    plt.colorbar(sc, ax=ax2, label='V1: Taxa de Presença')
    ax2.set_xlabel('t-SNE Dimensão 1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimensão 2', fontsize=11, fontweight='bold')
    ax2.set_title('Intensidade de Presença (V1) no espaço t-SNE', fontsize=12, fontweight='bold')

    plt.tight_layout()

    nome_arquivo = f"{ia}_{prompt_nome}_KMeans_tSNE.png"
    caminho_saida = pasta_graficos / nome_arquivo
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_saida

# =====================================================================
# 3. PROCESSAR TODOS OS CENÁRIOS
# =====================================================================
print("Visualizando K-Means via t-SNE para todos os cenários...\n")

arquivos_clusterizados = sorted(pasta_resultados.glob("Clusterizado_V3B*.csv"))
resumo = []

for i, arquivo_cluster in enumerate(arquivos_clusterizados, 1):
    nome = arquivo_cluster.name
    ia   = 'GOOGLE_AI' if 'GOOGLE_AI' in nome else ('OPENAI' if 'OPENAI' in nome else 'PERPLEXITY')
    prompt_nome = get_prompt_nome(nome)

    print(f"[{i}/9] {ia} - {prompt_nome}...", end=" ", flush=True)

    df_clusterizado  = pd.read_csv(arquivo_cluster)
    nome_original    = nome.replace('Clusterizado_V3B_', 'vetor_FINAL_')
    arquivo_original = pasta_processed / nome_original

    if arquivo_original.exists():
        df_original = pd.read_csv(arquivo_original)
        caminho = visualizar_tsne(df_original, df_clusterizado, ia, prompt_nome)
        print("OK")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Status': 'OK'})
    else:
        print("Arquivo original não encontrado")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Status': 'ERRO'})

print(f"\n{'='*60}")
print(pd.DataFrame(resumo).to_string(index=False))
print("\nGráficos salvos em: data/results/graficos/")
print("   Formato: IA_Prompt_KMeans_tSNE.png")
print(f"{'='*60}")
