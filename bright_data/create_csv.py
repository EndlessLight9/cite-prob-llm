import csv

headers = ["url", "prompt", "country", "index"]

prompt_list = ["Quais as melhores empresas de otimização de GEO/SEO no brasil?",
               "Quais são as melhores operadoras de celular no Brasil?",
               "Quais são as melhores marcas de fertilizante mineral no Brasil?",
               "Quais são as melhores lojas de roupas de frio em São Paulo?",
               "Quais são as melhores empresas de plataformas de marketplaces B2B e B2C no Brasil?"]

url= "https://google.com/aimode"
country = "BR"

for i, prompt_texto in enumerate(prompt_list):
    name = f"google_ai_prompt_{i+1}.csv"
    with open(name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for n in range(1, 1001):
            writer.writerow([url, prompt_list[i], country, n])