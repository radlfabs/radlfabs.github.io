import json
import requests

def make_request(data, url):
    response = requests.post(url, json=data)
    if response.status_code == 200:
        prediction = response.json()
        print(prediction)
    else:
        print(f'API Request Failed with Status Code: {response.status_code}')
        print(f'Response Content: {response.text}')


if __name__ == '__main__':
    # Define the URL of the Flask API
    url_simple = 'http://127.0.0.1:5000/predict'
    url_mapie = 'http://127.0.0.1:5000/predict-mapie'

    # load features for inference
    with open("data/inference_data.json") as f:
        data = json.load(f)

    make_request(data, url_simple)
    make_request(data, url_mapie)
