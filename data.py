import json
import os
import csv
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def process_qa_pairs(data):
	qa_pairs = []
	for item in data['data']:
		for paragraph in item['paragraphs']:
			for qa in paragraph['qas']:
				question = qa['question']
				answer = qa['answers'][0]['text'] if qa['answers'] else None
				qa_pairs.append((question, answer))
	return qa_pairs

def convert_json_to_csv(json_filename, csv_filename):
	# Open the JSON file for reading
	with open(json_filename, 'r') as file:
		# Load the JSON data
		data = json.load(file)

	# Extract questions and answers using the optimized function
	qa_pairs = process_qa_pairs(data)

	# Save to CSV
	with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(['Question', 'Answer'])
		csv_writer.writerows(qa_pairs)

	print(f'CSV file "{csv_filename}" created successfully!')

def merge_all_csv_files(input_folder, output_filename):
	# Get a list of all CSV files in the input folder
	csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

	# Ensure the output file does not exist yet
	if os.path.exists(output_filename):
		raise FileExistsError(f"The output file '{output_filename}' already exists. Please choose a different name.")

	# Merge CSV files into a single DataFrame
	dfs = [pd.read_csv(os.path.join(input_folder, csv_file)) for csv_file in csv_files]
	merged_df = pd.concat(dfs, ignore_index=True)

	# Save the merged DataFrame to a new CSV file
	merged_df.to_csv(output_filename, index=False, encoding='utf-8')

	print(f'Merged CSV file "{output_filename}" created successfully!')

# Get the current working directory
input_folder_path = os.getenv("DATASET_PATH")

# Function call with iteration over JSON files in the current working directory
for filename in os.listdir(input_folder_path):
	if filename.endswith('.json'):
		json_filepath = os.path.join(input_folder_path, filename)
		csv_filename = os.path.splitext(filename)[0] + '_output.csv'
		csv_filepath = os.path.join(input_folder_path, csv_filename)

		convert_json_to_csv(json_filepath, csv_filepath)

output_merged_filename = 'mashqa_merged_output_all.csv'
merge_all_csv_files(input_folder_path, output_merged_filename)
