import csv
from pathlib import Path

root = Path(__file__).parent.parent

googleaiurl = "https://google.com/aimode"
openaiurl = "https://chatgpt.com/"
perplexityurl = "https://www.perplexity.ai"
geminiurl = "https://gemini.google.com/"
bingurl = "https://copilot.microsoft.com/chats"

googleainame = "google_ai_prompt_test"
openainame = "open_ai_prompt_test"
perplexityname = "perplexity_prompt_test"
gemininame = "gemini_prompt_test"
bingname = "bing_prompt_test"

country = "BR"

headers = ["url", "prompt", "country", "index"]

prompt_list = ["Quais as melhores empresas de otimização de GEO/SEO no brasil?",
               "Quais são as melhores operadoras de celular no Brasil?",
               "Quais são as melhores marcas de fertilizante mineral no Brasil?",
               "Quais são as melhores lojas de roupas de frio em São Paulo?",
               "Quais são as melhores empresas de plataformas de marketplaces B2B e B2C no Brasil?"]

url= bingurl
name_start = bingname
country = "BR"

for i, prompt_texto in enumerate(prompt_list):
    name = f"{name_start}{i+1}.csv"
    path = root / name
    with open(path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for n in range(1, 6):
            writer.writerow([url, prompt_list[i] + f" [ignorar_id: {n:04d}]", country, n])