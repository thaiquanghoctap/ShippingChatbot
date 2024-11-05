import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import json

load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_KEY")
AWAN_KEY = os.getenv("AWAN_KEY")

pc = pinecone.Pinecone(api_key=PINECONE_KEY)


# Connect to pinecone database
index = pc.Index("tyme-test")

# Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Prompt template cho LLM
prompt_template = """
You are a helpful assistant that provides detailed information about product shipments.
The user is asking about a product shipment. Here are the product details:
Product Name: {order_name}
Price: {price} USD
Current Position: {current_pos}
Distance: {distance}

Please provide a clear and helpful response to the user based on these details.
"""

def awan(prompt, query):
    url = "https://api.awanllm.com/v1/chat/completions"

    payload = json.dumps({
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 1024,
        "stream": True 
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {AWAN_KEY}"
    }

    response = requests.post(url, headers=headers, data=payload, stream=True)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    result = ""
    for chunk in response.iter_lines():
        if chunk:
            decoded_chunk = chunk.decode('utf-8').replace("data: ", "")
            try:
                json_data = json.loads(decoded_chunk)
                content = json_data['choices'][0]['delta'].get('content', '')
                result += content
            except json.JSONDecodeError:
                print("Decoding error:", decoded_chunk)

    return result


def generate_response(query):
    query_embedding = embedding_model.encode(query).tolist()

    search_results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    
    if not search_results['matches']:
        return "Sorry, I couldn't find any relevant product information for your query."

    product_metadata = search_results['matches'][0]['metadata']
    
    prompt = prompt_template.format(
        order_name=product_metadata["order_name"],
        price=product_metadata["price"],
        current_pos=product_metadata["current_pos"],
        distance=product_metadata["distance"]
    )
    
    response = awan(prompt, query)

    return response


if __name__ == "__main__":
    query = "Gaming mouse"
    response = generate_response(query)
    print(response)
