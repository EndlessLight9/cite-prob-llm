from dotenv import load_dotenv
import requests
import os
from pathlib import Path

path = Path(__file__).parent.parent
csv_path = path / "perplexity_prompt_test3.csv"

load_dotenv()  # Load environment variables from .env file

url = "https://api.brightdata.com/datasets/v3/trigger"
headers = {
	"Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
}
params = {
	"dataset_id": os.getenv('DATASET_ID_PERPLEXITY'),
	"include_errors": "true",
}
files = {"data": ("data.csv", open(csv_path , "rb"), "text/csv")}
data = {
	"custom_output_fields": '["url","prompt","citations","timestamp","links_attached","answer_text", "sources"]' #"country", "search_sources"
}

response = requests.post(url, headers=headers, params=params, files=files, data=data)


if response.status_code == 200:
    try:
        response_data = response.json()
        print("Response JSON:", response_data)

    except requests.exceptions.JSONDecodeError:
        print("Error: Response content is not valid JSON.")
        print("Response content:", response.text)
else:
    print(f"Error: Received status code {response.status_code}")
    print("Response content:", response.text)