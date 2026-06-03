import pandas as pd
from pathlib import Path

root = Path(__file__).parent.parent
path_csv1 = root / 'notebook_data' / 'query_fan_out.csv'
path_csv2 = root / 'notebook_data' / 'search_and_cit_union_v1.csv'

df1 = pd.read_csv(path_csv1)
df2 = pd.read_csv(path_csv2)

df1.columns = df1.columns.str.strip()  
df2.columns = df2.columns.str.strip()

column_df1 = 'user_prompt'
column_df2 = 'prompt_query'

# --- ANÁLISE DO DF1 ---
if column_df1 in df1.columns and 'prompt_hash' in df1.columns:
    unique_total = df1[column_df1].nunique()
    
    # Agrupa pelo prompt e pelo hash, gerando uma tabela formatada
    count_prompts = df1[[column_df1, 'prompt_hash']].value_counts().reset_index(name='count')
    
    print(f"Unique values in {column_df1}: {unique_total}")
    print(f"Top prompts and hashes in {column_df1}:\n{count_prompts.head(30)}")
elif column_df1 in df1.columns:
    print("Atenção: 'prompt_hash' não encontrado no df1. Mostrando apenas os prompts:")
    print(df1[column_df1].value_counts().head(30))
else:
    print(f"Column {column_df1} not found in DataFrame 1.")

print("-" * 50)

# --- ANÁLISE DO DF2 ---
if column_df2 in df2.columns and 'prompt_hash' in df2.columns:
    unique_total = df2[column_df2].nunique()
    
    # Agrupa pelo prompt e pelo hash, gerando uma tabela formatada
    count_prompts = df2[[column_df2, 'prompt_hash']].value_counts().reset_index(name='count')

    print(f"\nUnique values in {column_df2}: {unique_total}")
    print(f"Top prompts and hashes in {column_df2}:\n{count_prompts.head(30)}")
elif column_df2 in df2.columns:
    print("\nAtenção: 'prompt_hash' não encontrado no df2. Mostrando apenas os prompts:")
    print(df2[column_df2].value_counts().head(30))
else:
    print(f"Column {column_df2} not found in DataFrame 2.")