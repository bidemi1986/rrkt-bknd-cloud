import requests
import json

def call_query(query, user_id=None, additional_params=None):
    """
    Call the query service with the given query and optional parameters.

    Args:
        query (str): The query string to send to the service.
        user_id (str, optional): The user ID to include in the payload.
        additional_params (dict, optional): Additional parameters to include in the payload.

    Returns:
        dict: The JSON response from the service, or an error message.
    """
    # Define the URL of the service
    url = "https://flask-app-685994944265.us-central1.run.app/"

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Define the payload
    payload = {
        "query": query,
        "user_id": user_id,
        "params": additional_params or {}
    }

    # Make the POST request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}



def call_greet(name="Guest"):
    """
    Call the greet service with an optional name parameter.

    Args:
        name (str, optional): The name to include in the query string. Defaults to "Guest".

    Returns:
        dict: The JSON response from the service, or an error message.
    """
    # Define the URL of the service
    url = f"https://flask-app-685994944265.us-central1.run.app/greet?name={name}"

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the GET request
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}
    

def call_vectorize(userId=None, additional_params=None):
    """
    Call the vectorization service with the given user ID and optional parameters.

    Args:
        user_id (str, optional): The user ID to include in the payload.
        additional_params (dict, optional): Additional parameters to include in the payload.

    Returns:
        dict: The JSON response from the service, or an error message.
    """
    # Define the URL of the service
    url = f"https://flask-app-685994944265.us-central1.run.app/vectorize"

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Define the payload
    payload = {
        "params": additional_params or {}
    }

    # Make the POST request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}
    

# Example usage
# response = call_query(
#     "What is the capital of France",
#     user_id="Cu3DdQMksLP7UEnZXmasiNlcEko1",
#     additional_params={"language": "en"}
# )

# greet
# response = call_greet()


# /vectorize/
response = call_vectorize( 
    userId="Cu3DdQMksLP7UEnZXmasiNlcEko1",
    additional_params={"userId": "Cu3DdQMksLP7UEnZXmasiNlcEko1"}
)
print("Response:", response)