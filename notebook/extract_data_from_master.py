import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
root = Path(__file__).parent.parent
path_master = root / 'notebook_data' / 'search_and_cit_union_v1.csv'

caminho_openai_master = path_master
df_openai = pd.read_csv(caminho_openai_master)

hashes_alvo = [
    os.getenv('hash_prompt_1'),
    os.getenv('hash_prompt_2'),
    os.getenv('hash_prompt_3')
]

df_openai_filtrado = df_openai[df_openai['prompt_hash'].isin(hashes_alvo)].copy()
df_openai_filtrado['url_hostname'] = df_openai_filtrado['url_hostname'].str.replace(r'^(https?://)?(www\.)?', '', regex=True)
df_openai_filtrado['SOURCE'] = 'OPENAI'

print(f"Total de linhas no dataset original: {len(df_openai)}")
print(f"Total de linhas extraídas após o filtro: {len(df_openai_filtrado)}")

contagem_hashes = df_openai_filtrado['prompt_hash'].value_counts()
print("\nContagem de cada hash no dataset filtrado:")
print(contagem_hashes)

caminho_salvamento = root / 'data' / 'openai_master.csv'
df_openai_filtrado.to_csv(caminho_salvamento, index=False)