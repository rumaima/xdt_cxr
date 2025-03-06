import os
import openai
import json
import pdb
from tqdm import tqdm

openai.api_key = ""
json_name = "Chexpert.json"

category_list = ["Atelectasis" , "Cardiomegaly",  "Consolidation",  "Edema",  "Enlarged Cardiomediastinum" , "Fracture" , "Lung Lesion" , "Lung_Opacity",  "No Finding" , "Pleural Effusion" , "Pleural Other" , "Pneumonia" , "Pneumothorax" , "Support_Devices"]

all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U']

for category in tqdm(category_list):

	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	prompts = []
	prompts.append("Describe pathalogically a photo of " + article + " " + category + " looks like ")
	prompts.append("Describe pathalogically" + article + " " + category + " as it would appear in a medical image")
	prompts.append("What does " + article + " " + category + " look like according to a medical expert?")
	prompts.append("How can you identify " + article + " "  + category + " in a chest xray image?")
	prompts.append("Describe the photo of "  + article + " "  + category +" from the perspective of a medical expert" )
	prompts.append("Describe a medical image of "  + article + " "  + category )

	all_result = []
	for curr_prompt in prompts:
		response = openai.Completion.create(
		    engine="gpt-3.5-turbo-instruct",
		    prompt=curr_prompt,
		    temperature=.99,
			max_tokens = 50,
			n=10,
			stop="."
		)

		for r in range(len(response["choices"])):
			result = response["choices"][r]["text"]
			all_result.append(result.replace("\n\n", "") + ".")

	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)