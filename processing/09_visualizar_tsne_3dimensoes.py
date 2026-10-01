import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURAÇÃO
# =====================================================================
path = Path(__file__).parent.parent
pasta_processed  = path / "data" / "processed"
pasta_resultados = path / "data" / "results" / "V3B_Neutro"
pasta_graficos   = path / "data" / "results" / "graficos" / "tridimensional"
pasta_graficos.mkdir(parents=True, exist_ok=True)

# Dois grupos de features — V1 sempre presente como âncora
GRUPOS = [
    {
        'nome': 'V1_V2_V3',
        'features': ['V1_taxa_presenca', 'V2_market_share', 'V3B_position_ignored'],
        'labels': ['V1: Presença', 'V2: Market Share', 'V3: Score Posicional'],
    },
    {
        'nome': 'V1_V4_V5',
        'features': ['V1_taxa_presenca', 'V4_taxa_conversao', 'V5_confiavel'],
        'labels': ['V1: Presença', 'V4: Conversão', 'V5: Estabilidade'],
    },
]

def get_prompt_nome(nome):
    if 'empresas de' in nome:
        return 'Prompt1_Empresas'
    if 'marcas d' in nome:
        return 'Prompt2_Fertilizante'
    if 'operador' in nome:
        return 'Prompt3_Operadoras'
    return 'PromptUnknown'


# =====================================================================
# 2. FUNÇÃO: VISUALIZAR K-MEANS DIRETAMENTE NO ESPAÇO DAS FEATURES (3D)
# =====================================================================
def visualizar_kmeans_3d(df_original, df_clusterizado, ia, prompt_nome, grupo):
    """
    Plota os pontos no espaço real das 3 features (normalizadas Z-score),
    que é exatamente o espaço onde o K-Means operou.
    Centroides são calculados como média dos pontos imputados+normalizados
    em cada cluster e marcados com ★.

    Os eixos correspondem diretamente às variáveis escolhidas,
    tornando a visualização interpretável e fiel ao modelo.
    """
    features = grupo['features']
    labels   = grupo['labels']

    features_disponiveis = [f for f in features if f in df_original.columns]
    if len(features_disponiveis) < 3:
        return None

    X        = df_original[features_disponiveis].copy()
    clusters = df_clusterizado['Cluster_ID'].values
    k        = len(np.unique(clusters))
    n_urls   = len(X)

    imputer   = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_imputed)

    # Calcular centroides no espaço normalizado para referência visual.
    np.array([
        X_scaled[clusters == c].mean(axis=0)
        for c in sorted(np.unique(clusters))
    ])

    colors = plt.cm.tab10(np.arange(k) / 10)

    fig = plt.figure(figsize=(28, 16))
    fig.suptitle(
        f'{ia} — {prompt_nome} | K-Means 3D — Espaço Real das Features ({grupo["nome"]})\n'
        f'Eixos: {labels[0]}  ×  {labels[1]}  ×  {labels[2]}\n'
        f'({n_urls} URLs, {k} clusters | valores normalizados Z-score)',
        fontsize=13, fontweight='bold'
    )

    for col_idx, azimute in enumerate([30, 135]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection='3d')
        ax.view_init(elev=20, azim=azimute)

        for idx_c, cluster_id in enumerate(sorted(np.unique(clusters))):
            mask = clusters == cluster_id
            ax.scatter(
                X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
                c=[colors[idx_c]],
                label=f'Cluster {cluster_id} ({mask.sum()} URLs)',
                s=40, alpha=0.75, edgecolors='black', linewidths=0.3
            )

        ax.set_xlabel(labels[0], fontsize=9, labelpad=6)
        ax.set_ylabel(labels[1], fontsize=9, labelpad=6)
        ax.set_zlabel(labels[2], fontsize=9, labelpad=6)
        ax.set_title(f'Ângulo {azimute}°', fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax.legend(loc='upper left', fontsize=8, markerscale=1.2)

    plt.tight_layout()

    nome_arquivo = f"{ia}_{prompt_nome}_{grupo['nome']}_KMeans3D.png"
    caminho_saida = pasta_graficos / nome_arquivo
    plt.savefig(caminho_saida, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    return caminho_saida


# =====================================================================
# 3. PROCESSAR TODOS OS CENÁRIOS E GRUPOS
# =====================================================================
print("Gerando visualizações K-Means 3D (espaço real) para todos os cenários...\n")

arquivos_clusterizados = sorted(pasta_resultados.glob("Clusterizado_V3B*.csv"))
resumo = []

total = len(arquivos_clusterizados) * len(GRUPOS)
contador = 0

for arquivo_cluster in arquivos_clusterizados:
    nome = arquivo_cluster.name
    ia   = 'GOOGLE_AI' if 'GOOGLE_AI' in nome else ('OPENAI' if 'OPENAI' in nome else 'PERPLEXITY')
    prompt_nome = get_prompt_nome(nome)

    df_clusterizado  = pd.read_csv(arquivo_cluster)
    nome_original    = nome.replace('Clusterizado_V3B_', 'vetor_FINAL_')
    arquivo_original = pasta_processed / nome_original

    if not arquivo_original.exists():
        for g in GRUPOS:
            contador += 1
            print(f"[{contador}/{total}] {ia} - {prompt_nome} ({g['nome']})... arquivo original não encontrado")
            resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Grupo': g['nome'], 'Status': 'ERRO'})
        continue

    df_original = pd.read_csv(arquivo_original)

    for g in GRUPOS:
        contador += 1
        print(f"[{contador}/{total}] {ia} - {prompt_nome} ({g['nome']})...", end=" ", flush=True)

        caminho = visualizar_kmeans_3d(df_original, df_clusterizado, ia, prompt_nome, g)

        if caminho:
            print("OK")
            resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Grupo': g['nome'], 'Status': 'OK'})
        else:
            print("Features insuficientes (menos de 3 colunas disponíveis)")
            resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Grupo': g['nome'], 'Status': 'SKIP'})

print(f"\n{'='*65}")
print(pd.DataFrame(resumo).to_string(index=False))
print("\nGráficos salvos em: data/results/graficos/tridimensional/")
print("   Formato: IA_Prompt_GRUPO_KMeans3D.png  (2 ângulos por grupo)")
print(f"{'='*65}")
