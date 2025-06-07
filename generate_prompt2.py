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
        "Compare the following sentence and identify only the parts related to color differences. "
        "Return a dictionary in the format: {changed: {obj: the changing object, ori: original_color, new: new_color}}. "
        "If no color difference return the object with new color, and left ori empty."
        "Keep the objest as simple as possiblet."
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


# Call it!
# text1 = "A cat with blue eye"
# text2 = "A cat with green eye"

# diff = compare_color_difference(text1, text2)
# print(diff)
results = {}

df = pd.read_csv(input_file_path)
granularity = df["granularity"].values
category = df["category"].values
low_color_idx = np.where((granularity == 'low') & (category == 'color'))[0]

for idx in low_color_idx:
    print(f"Comparing pair {idx+1}...")
    old_caption = df["input"].iloc[idx]
    caption = df["new_output"].iloc[idx]
    edited_caption = df["new_edit"].iloc[idx]
    comparison = compare_color_difference(edited_caption)
    results[int(idx)] = {
        "sentence": edited_caption,
        "difference": comparison
    }




# Save to a JSON file
with open("color_differences.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved to color_differences.json ")

