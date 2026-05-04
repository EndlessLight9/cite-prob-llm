from dotenv import load_dotenv
import requests
import os

load_dotenv()  # Load environment variables from .env file

url = "https://api.brightdata.com/datasets/v3/trigger"
headers = {
	"Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
}
params = {
	"dataset_id": os.getenv('DATASET_ID'),
	"include_errors": "true",
}
files = {"data": ("data.csv", open("/home/endless_light/cite-prob-llm/google_ai_prompt_3.csv", "rb"), "text/csv")}
data = {
	"custom_output_fields": '["url","prompt","citations","timestamp"]'
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