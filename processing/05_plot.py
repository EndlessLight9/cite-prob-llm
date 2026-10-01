import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS
# =====================================================================
path = Path(__file__).parent.parent
pasta_processed = path / "data" / "processed"
pasta_resultados = path / "data" / "results"

# Nova pasta para os gráficos de Top URLs
pasta_top_urls = pasta_resultados / "top_urls"
pasta_top_urls.mkdir(parents=True, exist_ok=True)

print("Gerando gráficos de Top 10 URLs para todos os atributos (V1-V5)...\n")

# Nomes dos atributos
atributos = {
    'V1_taxa_presenca': 'V1: Taxa de Presença',
    'V2_market_share': 'V2: Market Share',
    'V3': 'V3: Score Posicional ', # SO USAMOS V3B_position_ignored, renomeado para V3
    'V4_taxa_conversao': 'V4: Taxa Conversão ',
    'V5_confiavel': 'V5: Estabilidade '
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
# 2. FUNÇÃO: CRIAR GRÁFICOS DE BARRAS TOP 10
# =====================================================================
def criar_graficos_top10(df, ia, prompt_nome):
    """
    Cria 5 gráficos de barras horizontais (um para cada atributo V1-V5).
    Mostra o Top 10 domínios para cada cenário.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(32, 18))
    fig.suptitle(f'{ia} - {prompt_nome} | Top 10 Domínios por Atributo',
                 fontsize=18, fontweight='bold', y=0.98)

    axes = axes.flatten()
    
    # Identificar qual coluna V3 está presente (V3B ou V3A)
    coluna_v3 = 'V3B_position_ignored' #if 'V3B_position_ignored é a importante

    # Lista de tuplas com (Nome_Coluna, Título, Método_Top, Paleta)
    configs = [
        ('V1_taxa_presenca', atributos['V1_taxa_presenca'], 'nlargest', 'Blues_r'),
        ('V2_market_share', atributos['V2_market_share'], 'nlargest', 'Oranges_r'),
        (coluna_v3, atributos['V3'], 'nlargest', 'Greens_r'),
        ('V4_taxa_conversao', atributos['V4_taxa_conversao'], 'nlargest', 'Purples_r'),
        ('V5_confiavel' , 
         atributos['V5_confiavel'], 'nsmallest', 'Reds_r')
    ]

    for idx, (coluna, titulo, metodo, paleta) in enumerate(configs):
        ax = axes[idx]
        
        # Filtrar valores NaN (muito importante para V5_confiavel)
        df_valido = df.dropna(subset=[coluna])

        # Para V5, manter apenas desvios estritamente positivos.
        if coluna == 'V5_confiavel':
            df_valido = df_valido[df_valido[coluna] > 0]
        
        if len(df_valido) == 0:
            ax.text(0.5, 0.5, 'Dados insuficientes/NaN', ha='center', va='center', fontsize=12)
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            continue

        # Selecionar Top 10 (Maior para V1-V4, Menor para V5)
        if metodo == 'nlargest':
            df_top = df_valido.nlargest(10, coluna)
        else:
            df_top = df_valido.nsmallest(10, coluna)

        # Plotar Gráfico de Barras Horizontal
        sns.barplot(data=df_top, x=coluna, y='url_hostname', ax=ax, palette=paleta)

        # Configurar eixos e estética
        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel('Valor do Atributo', fontsize=11)
        ax.set_ylabel('')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adicionar os valores numéricos na ponta das barras
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + (width * 0.02), p.get_y() + p.get_height() / 2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)

    # Remover o 6º subplot (deixar em branco)
    fig.delaxes(axes[5])

    plt.tight_layout(pad=3.0)

    # Salvar figura
    nome_arquivo = f"{ia}_{prompt_nome}_Top10_URLs.png"
    caminho_saida = pasta_top_urls / nome_arquivo
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return caminho_saida

# =====================================================================
# 3. PROCESSAR TODOS OS CENÁRIOS
# =====================================================================
arquivos_vetores = sorted(pasta_processed.glob("vetor_FINAL_*.csv"))
contador = 0
resumo = []

for arquivo_vetor in arquivos_vetores:
    contador += 1

    # Extrair IA e prompt
    nome = arquivo_vetor.name
    ia = 'GOOGLE_AI' if 'GOOGLE_AI' in nome else ('OPENAI' if 'OPENAI' in nome else 'PERPLEXITY')
    prompt_nome = get_prompt_nome(nome)

    print(f"[{contador}/{len(arquivos_vetores)}] {ia} - {prompt_nome}...", end=" ")

    # Carregar dados
    df_original = pd.read_csv(arquivo_vetor)

    if len(df_original) > 0:
        # Gerar os dashboards de Top 10
        caminho = criar_graficos_top10(df_original, ia, prompt_nome)
        print("Salvo.")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Status': 'OK'})
    else:
        print("Arquivo vazio.")
        resumo.append({'IA': ia, 'Prompt': prompt_nome, 'Status': 'VAZIO'})

# =====================================================================
# 4. RESUMO FINAL
# =====================================================================
print(f"\n{'='*70}")
print("GRÁFICOS DE TOP URLs GERADOS COM SUCESSO")
print(f"{'='*70}\n")

df_resumo = pd.DataFrame(resumo)
print(df_resumo.to_string(index=False))

print(f"\n{'='*70}")
print("Localização: data/results/top_urls/")
print(f"Total de painéis: {len(list(pasta_top_urls.glob('*Top10_URLs*.png')))}")
print(f"{'='*70}\n")