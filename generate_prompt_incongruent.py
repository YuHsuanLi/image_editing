import json
import requests
import numpy as np
import pandas as pd
import os
from langchain.schema import (
    HumanMessage, 
    SystemMessage
)

from openai import OpenAI
import openai
# openai api here
BASE_URL = ""
DEPLOYMENT_NAME = "" 
## input your API key here
API_KEY = "" 
input_file_path = "./Data/High-Low/high_low_final_w_idx.csv"



def compare_color_difference(text):
    sys_message = (
        "Given the following sentence, identify the part that has changed and should be edited in the original image."
        "The answer refers to the element that is present in the original image."
        "Return a dictionary in the format: {changed: {obj: the changing object}}. "
        "Keep the objest as simple as possible."
        "only show one biggest changing part."
        "in pure JSON format without code block."
    )

    request = f"""
    Sentence : "{text}"
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": request}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content



results = {}
with open(input_file_path, 'r') as f:
    data = [json.loads(line) for line in f]

i = 0
for idx in range(len(data)):
    print(f"Comparing pair {idx+1}...")
    edited_caption = data[idx]['incongruent_edit']
    comparison = compare_color_difference(edited_caption)
    results[int(idx)] = {
        "sentence": edited_caption,
        "difference": comparison
    }
    i += 1


# Save to a JSON file
with open("incongruent2.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved to color_differences.json ")

