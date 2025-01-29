import json
import pytest
import os
import re
import asyncio
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# import import_ipynb
# import src.scraper
import base64
import os
import json
from src.agents import WebAgent
from src.custom_types import AgentState
from src.scraper import Scraper,SCRAPER_PROMPT

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
    
""" Load JSON data from a given file path.
    Parameters: str: The path to the JSON file to be loaded.
    Returns: dict: The JSON data loaded from the file.
"""
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


""" Calculates the accuracy of the predited price which is scraped from the image.
    
    Parameters:

    predicted: dict: predicted output prices
    expected_output_path: str: The path to the JSON file to be loaded which contains expected JSON output fro each input image.
    
    Returns: float: Return the accuarcy of the predicticted outcome.
"""
def check_prices_accuracy(predicted, expected_output_path):
    expected = load_json(expected_output_path)
    predicted_prices = {entry.get("price") for entry in predicted.get("prices", [])}
    expected_prices = {entry.get("price") for entry in expected.get("prices", [])}
    matching_prices = predicted_prices.intersection(expected_prices)
    accuracy = len(matching_prices) / max(len(expected_prices), len(predicted_prices)) if expected_prices else 0.0
    return accuracy


images = [
    'tests/scraper_imgs/image.png',
    'tests/scraper_imgs/BILL.png',
    'tests/scraper_imgs/commerce.png',
    'tests/scraper_imgs/image2.png',
    'tests/scraper_imgs/internet_1.png',
    'tests/scraper_imgs/Netflix.png'
]


# Expected output JSON files for each image
expected_json_files = [
    'tests/scraper_outputs/image.json',
    'tests/scraper_outputs/BILL.json',
    'tests/scraper_outputs/commerce.json',
    'tests/scraper_outputs/image2.json',
    'tests/scraper_outputs/internet_1.json',
    'tests/scraper_outputs/Netflix.json'
]

# agent = WebAgent()
# llm = agent.llm

@pytest.mark.asyncio
async def test_scraper(llm):

    scraper = Scraper(
    mm_llm= ChatGroq(model=llm),
    prompt = SCRAPER_PROMPT
)  


    for image_path, expected_file in zip(images, expected_json_files):
        try:
            imgs=encode_image(image_path=image_path)
            state = AgentState(img=imgs)
            # print("*****")
            # print(state)
            # print("*****")
            result_state = await scraper.run(state)
            observation = result_state.get("observation")
            if observation:
                predicted = json.loads(observation)
                accuracy = check_prices_accuracy(predicted, expected_file)
                print(f"Accuracy for {image_path}: {accuracy:.2%}")
            
            else:
                print(f"No observation found for {image_path}")


        except Exception as e:
            print(f"Exception occurred for {image_path}: {e}")
            continue
            
   
            
if __name__ == "__main__":
    llm = "llama-3.2-90b-vision-preview"
    asyncio.run(test_scraper(llm))