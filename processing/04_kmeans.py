import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS
# =====================================================================
path = Path(__file__).parent.parent
pasta_processed = path / "data" / "processed"

pasta_resultados_A = path / "data" / "results" / "V3A_Penalizado"
pasta_resultados_B = path / "data" / "results" / "V3B_Neutro"

pasta_resultados_A.mkdir(parents=True, exist_ok=True)
pasta_resultados_B.mkdir(parents=True, exist_ok=True)

print("Iniciando K-Means Clustering com Elbow Method...")

arquivos_vetores = sorted(pasta_processed.glob("vetor_FINAL_*.csv"))

if not arquivos_vetores:
    print("Nenhum ficheiro encontrado na pasta 'processed'.")
    exit()

# =====================================================================
# 2. FUNÇÃO AUXILIAR: ENCONTRAR K ÓTIMO
# =====================================================================
def encontrar_k_otimo(X_scaled, max_k=10):
    silhuetas = []
    inercias = []
    K_range = range(2, min(max_k, len(X_scaled) - 1))

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500)
        labels = km.fit_predict(X_scaled)

        silhueta = silhouette_score(X_scaled, labels)
        silhuetas.append(silhueta)
        inercias.append(km.inertia_)

    k_otimo = K_range[np.argmax(silhuetas)]
    return k_otimo, K_range, inercias, silhuetas

# =====================================================================
# 3. HIPÓTESES A/B
# =====================================================================
hipoteses = [
    {
        "nome": "Hipótese A (O Carrossel Importa)",
        "features": ['V1_taxa_presenca', 'V2_market_share', 'V3A_position_important', 'V4_taxa_conversao', 'V5_confiavel'],
        "pasta_destino": pasta_resultados_A,
        "sufixo": "Clusterizado_V3A"
    },
    {
        "nome": "Hipótese B (Estar no Backend Basta)",
        "features": ['V1_taxa_presenca', 'V2_market_share', 'V3B_position_ignored', 'V4_taxa_conversao', 'V5_confiavel'],
        "pasta_destino": pasta_resultados_B,
        "sufixo": "Clusterizado_V3B"
    }
]

# =====================================================================
# 4. PROCESSAMENTO PRINCIPAL
# =====================================================================
for arquivo in arquivos_vetores:
    print(f"\n Processando: {arquivo.name}")
    df_original = pd.read_csv(arquivo)

    if len(df_original) < 4:
        print("   Ignorado: Menos de 4 URLs neste cenário.")
        continue

    for hip in hipoteses:
        df_ml = df_original.copy()
        X = df_ml[hip['features']].copy()

        # --- PASSO A: IMPUTAÇÃO DOS NaN (V5_confiavel) COM MEDIANA ---
        # NaN em V5_confiavel = URL sem dados confiáveis de posição (CASO_2 ou CASO_3)
        # Estratégia: imputar com a mediana do cenário
        # Motivo: não premiar (0) nem penalizar (máximo) URLs desconhecidas
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # --- PASSO B: PADRONIZAÇÃO (Z-SCORE) ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Registrar quantos NaN foram imputados e qual valor foi usado
        n_nan = X[hip['features'][-1]].isna().sum()
        valor_imputado = imputer.statistics_[-1]  # mediana calculada para V5_confiavel

        # --- PASSO B: ENCONTRAR K ÓTIMO ---
        k_otimo, K_range, inercias, silhuetas = encontrar_k_otimo(X_scaled, max_k=10)

        # --- PASSO C: CLUSTERIZAÇÃO COM K ÓTIMO ---
        kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=50, max_iter=500, algorithm='lloyd')
        df_ml['Cluster_ID'] = kmeans.fit_predict(X_scaled)

        # --- PASSO D: MÉTRICAS DE QUALIDADE ---
        labels = df_ml['Cluster_ID'].values
        inertia = kmeans.inertia_
        silhueta = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

        # --- PASSO E: ORDENAÇÃO E EXPORTAÇÃO ---
        df_ml = df_ml.sort_values(by=['Cluster_ID', 'V1_taxa_presenca'], ascending=[True, False])

        nome_saida = arquivo.name.replace("vetor_FINAL", hip['sufixo'])
        caminho_saida = hip['pasta_destino'] / nome_saida
        df_ml.to_csv(caminho_saida, index=False)

        # Salvar scaler para produção
        caminho_scaler = caminho_saida.with_suffix('.pkl')
        joblib.dump(scaler, caminho_scaler)

        # Salvar métricas
        caminho_metricas = caminho_saida.with_suffix('.txt')
        with open(caminho_metricas, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write("K-Means Clustering Metrics\n")
            f.write(f"{'='*60}\n")
            f.write(f"Hipótese: {hip['nome']}\n")
            f.write(f"Cenário: {arquivo.name}\n\n")
            f.write("Otimização do K:\n")
            f.write(f"  K ótimo encontrado: {k_otimo}\n")
            f.write(f"  Range testado: {list(K_range)}\n")
            f.write(f"  Silhuetas por K: {[f'{s:.3f}' for s in silhuetas]}\n\n")
            f.write("Tratamento do V5_confiavel (NaN):\n")
            f.write("  Estratégia: imputação pela mediana do cenário\n")
            f.write(f"  NaN imputados: {n_nan} URLs\n")
            f.write(f"  Valor da mediana usada: {valor_imputado:.4f}\n\n")
            f.write("Métricas de Qualidade:\n")
            f.write(f"  Inércia: {inertia:.2f}\n")
            f.write(f"  Silhueta Score: {silhueta:.3f} ([-1,1]: mais perto de 1 = melhor)\n")
            f.write(f"  Davies-Bouldin Index: {davies_bouldin:.3f} (menor = melhor)\n")
            f.write(f"  Calinski-Harabasz Index: {calinski_harabasz:.2f} (maior = melhor)\n\n")
            f.write("Distribuição dos Clusters:\n")
            cluster_counts = df_ml['Cluster_ID'].value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                f.write(f"  Cluster {cluster_id}: {count} URLs\n")

        print(f"  {hip['nome']}")
        print(f"     K ótimo: {k_otimo} | Silhueta: {silhueta:.3f} | Davies-Bouldin: {davies_bouldin:.3f}")

print("\nK-Means clustering concluído com sucesso!")