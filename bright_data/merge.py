import pandas as pd
import json
import re
from pathlib import Path

# ==========================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ==========================================
root = Path(__file__).parent.parent
pasta_dados = root / 'data'

# ==========================================
# 2. LISTA DOS LOTES EXTRAÍDOS
# ==========================================
arquivos_para_processar = [
    # ---- GOOGLE AI ----
    {'arquivo': 'googleai_prompt_1.csv', 'source': 'GOOGLE_AI', 'brand_name': 'Ranqia', 'brand_id': 28},
    {'arquivo': 'googleai_prompt_2.csv', 'source': 'GOOGLE_AI', 'brand_name': 'Fluke', 'brand_id': 19},
    {'arquivo': 'googleai_prompt_3.csv', 'source': 'GOOGLE_AI', 'brand_name': 'AgroMinas', 'brand_id': 21},
    
    # ---- OPENAI ----
    {'arquivo': 'openai_prompt_1.csv', 'source': 'OPENAI', 'brand_name': 'Ranqia', 'brand_id': 28},
    {'arquivo': 'openai_prompt_2.csv', 'source': 'OPENAI', 'brand_name': 'Fluke', 'brand_id': 19},
    {'arquivo': 'openai_prompt_3.csv', 'source': 'OPENAI', 'brand_name': 'AgroMinas', 'brand_id': 21},
    
    # ---- PERPLEXITY ----
    {'arquivo': 'perplexity_prompt_1.csv', 'source': 'PERPLEXITY', 'brand_name': 'Ranqia', 'brand_id': 28},
    {'arquivo': 'perplexity_prompt_2.csv', 'source': 'PERPLEXITY', 'brand_name': 'Fluke', 'brand_id': 19},
    {'arquivo': 'perplexity_prompt_3.csv', 'source': 'PERPLEXITY', 'brand_name': 'AgroMinas', 'brand_id': 21},
]

# ==========================================
# 3. FUNÇÕES AUXILIARES DE EXTRAÇÃO E LIMPEZA
# ==========================================
def extrair_dominio(url):
    if not isinstance(url, str): return None
    # Limpa http, https, www e remove eventuais barras finais
    limpo = re.sub(r'^(https?://)?(www\.)?', '', url)
    limpo = limpo.split('/')[0]
    return limpo

def extrair_marca_da_url(dominio):
    if not dominio: 
        return []
    partes = dominio.split('.')
    
    # Ignora subdomínios comuns
    subdominios_ignorados = ['blog', 'loja', 'app', 'pt', 'en', 'br', 'news', 'web','agencies']
    if partes[0] in subdominios_ignorados and len(partes) > 1:
        nucleo = partes[1].lower()
    else:
        nucleo = partes[0].lower()
        
    variacoes = [nucleo] # Ex: 'upsendbrasil'
    
    # Dicionário inteligente de sufixos comuns no mercado corporativo
    sufixos_comuns = [
        'brasil', 'tech', 'martin', 'fertil', 'mkt', 'digital', 
        'organica', 'online', 'solo', 'agro', 'minas', 'bh'
    ]
    
    for sufixo in sufixos_comuns:
        # Se a palavra terminar com o sufixo (evitando separar se a palavra for exata e unicamente o sufixo)
        if nucleo.endswith(sufixo) and nucleo != sufixo:
            prefixo = nucleo[:-len(sufixo)]
            variacoes.append(f"{prefixo} {sufixo}") # Adiciona a versão separada: 'upsend brasil'
            break
            
    return variacoes

# ==========================================
# 4. FUNÇÃO PRINCIPAL DE PROCESSAMENTO E FUNIL
# ==========================================
def processar_lote_bruto(info):
    caminho_arquivo = pasta_dados / info['arquivo']
    
    if not caminho_arquivo.exists():
        print(f"⏳ A aguardar ficheiro: {info['arquivo']}")
        return pd.DataFrame()
        
    print(f"✅ A processar: {info['arquivo']} ({info['source']}) -> Alvo Principal: {info['brand_name']}")
    df = pd.read_csv(caminho_arquivo)
    
    # Cria identificador único para cada teste/sessão
    df['test_id'] = range(1, len(df) + 1)
    
    linhas_processadas = []

    # Iteração sobre cada teste para construir o dossiê de URLs
    for index, row in df.iterrows():
        
        # Carregamento seguro das estruturas JSON
        try: citations = json.loads(row['citations']) if pd.notnull(row['citations']) else []
        except: citations = []
            
        try: links_attached = json.loads(row['links_attached']) if pd.notnull(row['links_attached']) else []
        except: links_attached = []

        dominios_map = {}

        # -----------------------------------------------------
        # PASSO A: Análise dos Bastidores (citations)
        # -----------------------------------------------------
        for c in citations:
            dominio = extrair_dominio(c.get('url') or c.get('domain'))
            if not dominio: continue

            # Regras específicas por Arquitetura de IA:
            if info['source'] in ['GOOGLE_AI', 'OPENAI']:
                status_atual = c.get('cited', False)
            else:
                # PERPLEXITY: a mera presença no array significa recuperação primária pelo RAG
                status_atual = True

            if dominio not in dominios_map:
                dominios_map[dominio] = {'cited_status': status_atual, 'posicoes': []}
            else:
                dominios_map[dominio]['cited_status'] = dominios_map[dominio]['cited_status'] or status_atual

        # -----------------------------------------------------
        # PASSO B: Análise da Interface/Visibilidade (links_attached)
        # -----------------------------------------------------
        for l in links_attached:
            dominio = extrair_dominio(l.get('url'))
            if not dominio: continue

            if dominio not in dominios_map:
                dominios_map[dominio] = {'cited_status': False, 'posicoes': []}

            pos = l.get('position')
            if pos is not None:
                dominios_map[dominio]['posicoes'].append(pos)

        # -----------------------------------------------------
        # PASSO C & D: Construção do Funil de Avaliação e Conversão (Y)
        # -----------------------------------------------------
        texto_resposta = str(row['answer_text']).lower()

        for dominio, dados in dominios_map.items():
            
            # Obtém a lista de variações (ex: ['colinatech', 'colina tech'])
            variacoes_marca = extrair_marca_da_url(dominio)
            
            # Verifica se QUALQUER variação ocorreu dentro do texto gerado
            url_mencionada = 0
            for var in variacoes_marca:
                if var and var in texto_resposta:
                    url_mencionada = 1
                    break

            linhas_processadas.append({
                'test_id': row['test_id'],
                'prompt_query': row['prompt'],
                'url_hostname': dominio,
                'target_brand': str(variacoes_marca), # Registo para auditoria
                
                # --- VARIÁVEIS DEPENDENTES DE GEO ---
                'cited_status': dados['cited_status'],                                  
                'link_mentioned_in_response': 1 if len(dados['posicoes']) > 0 else 0,   
                'brand_mentioned_in_response': url_mencionada,                          
                
                # --- MÉTRICAS AUXILIARES ---
                'posicoes_links': str(dados['posicoes']),
                'frequencia_citacao': len(dados['posicoes']),
                'searched_at': row['timestamp'],
                'answer_text': row['answer_text'],
                'source': info['source'],
                'brand_name': info['brand_name'],
                'brand_id': info['brand_id']
            })

    # Construção do DataFrame consolidado do lote
    df_final = pd.DataFrame(linhas_processadas)
    
    # Tratamento temporal para formato padrão analítico
    if 'searched_at' in df_final.columns:
        df_final['searched_at'] = pd.to_datetime(df_final['searched_at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    return df_final

# ==========================================
# 5. EXECUÇÃO E CONSOLIDAÇÃO GERAL (MERGE)
# ==========================================
lista_dfs_processados = []

for item in arquivos_para_processar:
    df_limpo = processar_lote_bruto(item)
    if not df_limpo.empty:
        lista_dfs_processados.append(df_limpo)

if lista_dfs_processados:
    df_master = pd.concat(lista_dfs_processados, ignore_index=True)
    
    # Exportação do Master Dataset
    caminho_salvamento = pasta_dados / 'master_dataset_tcc.csv'
    df_master.to_csv(caminho_salvamento, index=False)
    
    print("\n🚀 Pipeline analítico concluído com sucesso!")
    print(f"📊 O Master Dataset contém {len(df_master)} registos de domínios consolidados.")
else:
    print("\nNenhum ficheiro processado. Verifique os nomes e a estrutura dos diretórios.")