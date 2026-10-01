import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================================================
# 1. CONFIGURAÇÃO
# =====================================================================
path = Path(__file__).parent.parent
pasta_processed    = path / "data" / "processed"
pasta_resultados   = path / "data" / "results" / "V3B_Neutro"
pasta_graficos     = path / "data" / "results" / "graficos" / "Kmeans"
pasta_graficos.mkdir(parents=True, exist_ok=True)

FEATURES = ['V1_taxa_presenca', 'V2_market_share', 'V3B_position_ignored',
            'V4_taxa_conversao', 'V5_confiavel']

def get_prompt_nome(nome):
    if 'empresas de' in nome:
        return 'Prompt1_Empresas'
    if 'marcas d' in nome:
        return 'Prompt2_Marcas'
    if 'operador' in nome:
        return 'Prompt3_Operadoras'
    return 'PromptUnknown'

# =====================================================================
# 2. FUNÇÃO: VISUALIZAR K-MEANS VIA PCA
# =====================================================================
def visualizar_kmeans(df_original, df_clusterizado, ia, prompt_nome):
    """
    Reduz as 5 dimensões (V1-V5) para 2D via PCA e plota os clusters.

    PCA (Principal Component Analysis):
    - Encontra as direções de maior variância nos dados
    - PC1 = direção que explica mais variância
    - PC2 = direção perpendicular a PC1 que explica a segunda maior variância
    - Juntos, PC1+PC2 resumem o máximo de informação possível em 2D
    """

    # --- PREPARAR DADOS ---
    X = df_original[FEATURES].copy()
    clusters = df_clusterizado['Cluster_ID'].values
    k = len(np.unique(clusters))

    # Imputar NaN com mediana (mesmo processo do K-Means)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # --- APLICAR PCA ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    variancia_pc1 = pca.explained_variance_ratio_[0] * 100
    variancia_pc2 = pca.explained_variance_ratio_[1] * 100
    variancia_total = variancia_pc1 + variancia_pc2

    # --- CRIAR FIGURA (2 subgráficos) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'{ia} — {prompt_nome} | K-Means via PCA\n'
                 f'(PC1={variancia_pc1:.1f}% + PC2={variancia_pc2:.1f}% = {variancia_total:.1f}% da variância)',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.arange(k) / 10)

    # ====== GRÁFICO 1: SCATTER PCA - todos os pontos ======
    ax1 = axes[0]
    for cluster_id in sorted(np.unique(clusters)):
        mask = clusters == cluster_id
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[colors[cluster_id]], label=f'Cluster {cluster_id} ({mask.sum()} URLs)',
                    s=60, alpha=0.7, edgecolors='black', linewidth=0.3)

    ax1.set_xlabel(f'PC1 ({variancia_pc1:.1f}% da variância)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({variancia_pc2:.1f}% da variância)', fontsize=11, fontweight='bold')
    ax1.set_title('Clusters no espaço PCA', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # Destacar os top 5 URLs mais presentes com label
    top5_idx = df_original['V1_taxa_presenca'].nlargest(5).index
    for idx in top5_idx:
        pos = np.where(df_original.index == idx)[0]
        if len(pos) > 0:
            i = pos[0]
            ax1.annotate(df_original.loc[idx, 'url_hostname'],
                         (X_2d[i, 0], X_2d[i, 1]),
                         fontsize=7, ha='center', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

    # ====== GRÁFICO 2: CONTRIBUIÇÃO DE CADA FEATURE NOS PCs (LOADINGS) ======
    ax2 = axes[1]
    loadings = pca.components_.T  # shape (n_features, 2)
    feature_labels = ['V1\nPresença', 'V2\nMarket Share', 'V3\nPosição',
                      'V4\nConversão', 'V5\nEstabilidade']

    x_pos = np.arange(len(feature_labels))
    width = 0.35

    ax2.bar(x_pos - width / 2, loadings[:, 0], width,
            label=f'PC1 ({variancia_pc1:.1f}%)', color='steelblue', edgecolor='black', linewidth=0.8)
    ax2.bar(x_pos + width / 2, loadings[:, 1], width,
            label=f'PC2 ({variancia_pc2:.1f}%)', color='coral', edgecolor='black', linewidth=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_labels, fontsize=10)
    ax2.set_ylabel('Contribuição (Loading)', fontsize=11, fontweight='bold')
    ax2.set_title('Quanto cada atributo contribui para PC1 e PC2', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    nome_arquivo = f"{ia}_{prompt_nome}_KMeans_PCA.png"
    caminho_saida = pasta_graficos / nome_arquivo
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_saida, variancia_total

# =====================================================================
# 3. PROCESSAR TODOS OS CENÁRIOS
# =====================================================================
print("Visualizando K-Means via PCA para todos os cenários...\n")

arquivos_clusterizados = sorted(pasta_resultados.glob("Clusterizado_V3B*.csv"))
resumo = []

for i, arquivo_cluster in enumerate(arquivos_clusterizados, 1):
    nome = arquivo_cluster.name
    ia = 'GOOGLE_AI' if 'GOOGLE_AI' in nome else ('OPENAI' if 'OPENAI' in nome else 'PERPLEXITY')
    prompt_nome = get_prompt_nome(nome)

    print(f"[{i}/9] {ia} - {prompt_nome}...", end=" ")

    df_clusterizado = pd.read_csv(arquivo_cluster)
    nome_original   = nome.replace('Clusterizado_V3B_', 'vetor_FINAL_')
    arquivo_original = pasta_processed / nome_original

    if arquivo_original.exists():
        df_original = pd.read_csv(arquivo_original)
        caminho, variancia = visualizar_kmeans(df_original, df_clusterizado, ia, prompt_nome)
        print(f"Variância preservada: {variancia:.1f}%")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Variância PCA': f"{variancia:.1f}%"})
    else:
        print("Arquivo original não encontrado")

print(f"\n{'='*60}")
print(pd.DataFrame(resumo).to_string(index=False))
print("\nGráficos salvos em: data/results/graficos/")
print(f"{'='*60}")
