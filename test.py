import requests
import json

# Define the URL of your FastAPI app
url = "http://127.0.0.1:8000/predict"  # Change if you are using a different host/port

# Create a dictionary with the data to send to the API
data = {
    "post_content": "I got:\n\nMaths - 9\nPhysics-9 \nChemistry- 9 \nBiology- 9\nEnglish language- 9 \nEnglish literature- 8 \nFrench- 9\nPsychology- 9 \nReligious studies- 9\nHistory- 8\nDesign technology- 8\n\nIf you have any questions/ seeking advice feel free to ask xx",
    "post_title": "I got 99999999888 at GCSE, ask me anything ‼️",
    "category": "GCSE"
}

# Convert the dictionary to JSON format
headers = {'Content-Type': 'application/json'}

# Make the POST request to the FastAPI endpoint
response = requests.post(url, data=json.dumps(data), headers=headers)

# Print out the response
if response.status_code == 200:
    print("Success!")
    print(f"Response: {response.json()}")
else:
    print(f"Failed with status code {response.status_code}")
    print(f"Error message: {response.text}")
