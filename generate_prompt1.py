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


client = openai.OpenAI(
  api_key=API_KEY,
  base_url="",
)

def compare_color_difference(text1, text2):
    sys_message = (
        "Compare the following two sentences and identify only the parts related to color differences. "
        "Return a dictionary in the format: {changed: {obj: the changing object, ori: original_color, new: new_color}}. "
        "If no color difference is found, return an empty dictionary."
        "in pure JSON format without code block."
    )

    request = f"""
    Sentence 1: "{text1}"
    Sentence 2: "{text2}"
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

df = pd.read_csv(input_file_path)
granularity = df["granularity"].values
category = df["category"].values
# granularity=low and category=color
low_color_idx = np.where((granularity == 'low') & (category == 'color'))[0]
i = 0
for idx in low_color_idx:
    print(f"Comparing pair {idx+1}...")
    old_caption = df["input"].iloc[idx]
    caption = df["new_output"].iloc[idx]
    comparison = compare_color_difference(old_caption, caption)
    results[int(idx)] = {
        "sentence_1": old_caption,
        "sentence_2": caption,
        "difference": comparison
    }
   
# Save to a JSON file
with open("color_differences.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved to color_differences.json ")

