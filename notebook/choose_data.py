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

if column_df1 in df1.columns:
    unique_total = df1[column_df1].nunique()
    count_prompts = df1[column_df1].value_counts()
    #sorting so only lines that have "ranqia_research" in the "brand_name" column are shown
    #count_prompts = count_prompts[df1.loc[df1['brand_name'].str.contains('ranqia-research'), column_df1].value_counts().index]
    print(f"Unique values in {column_df1}: {unique_total}")
    print(f"Top prompts in {column_df1}:\n{count_prompts.head(30)}")
else:
    print(f"Column {column_df1} not found in DataFrame.")

if column_df2 in df2.columns:
    unique_total = df2[column_df2].nunique()
    count_prompts = df2[column_df2].value_counts()
    #sorting so only lines that have "ranqia_research" in the "brand_name" column are shown
    #count_prompts = count_prompts[df2.loc[df2['brand_name'].str.contains('ranqia-research'), column_df2].value_counts().index]

    print(f"Unique values in {column_df2}: {unique_total}")
    print(f"Top prompts in {column_df2}:\n{count_prompts.head(30)}")
else:
    print(f"Column {column_df2} not found in DataFrame.")   
