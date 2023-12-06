"""Program to get insights and metrics for the results of the experiments that are run for the GraphQL environment.

The experiments script gives the data in JSON format. To understand the performance of different LLMs, the JSON data is converted to CSV.
Then the values from the CSV are taken compute the percentageentage of correct results
"""
import json
import csv
import math

# Common configuration for running the functions
FILE_NAME = "ic_graphql_multiturn_gpt-4_7_turns"
FILE_PATH = f"./logs/experiments/{FILE_NAME}.json"


def generate_csv_results(json_file_path: str, output_path: str, csv_fields: list):
    json_result = {}

    with open(json_file_path) as f:
        json_result = json.load(f)

    csv_list = []

    for _, value in json_result.items():
        summary = value["summary"]
        max_reward = summary["max_reward"]
        max_query_reward = summary["max_query_reward"]
        turns_taken = summary["turns_taken"]
        turns_max = summary["turns_max"]
        hardness = value["hardness"]
        correct = int(max_reward) == 1
        csv_list.append({
            "Max Reward": max_reward,
            "Max Query Reward": max_query_reward,
            "Turns Taken": turns_taken,
            "Max Turns Allowed": turns_max,
            "Hardness": hardness,
            "Correct": correct
        })

    with open(output_path, 'w', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = csv_fields)
        writer.writeheader()
        writer.writerows(csv_list)

    print("Successfully generated the CSV file!!")

OUTPUT_PATH = f"./{FILE_NAME}.csv"
CSV_FIELDS = ["Max Reward", "Max Query Reward", "Turns Taken", "Max Turns Allowed", "Hardness", "Correct"]
# generate_csv_results(FILE_PATH, OUTPUT_PATH, CSV_FIELDS)

# Distribution of easy, medium, hard questions in the dataset
EASY_TOTAL = 12
MEDIUM_TOTAL = 11
HARD_TOTAL = 7
TOTAL_QUERIES = EASY_TOTAL + MEDIUM_TOTAL + HARD_TOTAL


def compute_correct_percentage_results(json_file_path: str):
    json_result = {}

    with open(json_file_path) as f:
        json_result = json.load(f)

    # Initialise results
    easy_correct = 0
    medium_correct = 0
    hard_correct = 0

    for _, value in json_result.items():
        summary = value["summary"]
        max_reward = summary["max_reward"]
        is_correct = int(max_reward) == 1

        if is_correct:
            hardness = value["hardness"]

            if hardness == "easy":
                easy_correct += 1
            elif hardness == "medium":
                medium_correct += 1
            else:
                hard_correct += 1

    easy_correct_percentage = math.floor((easy_correct / TOTAL_QUERIES) * 100)
    medium_correct_percentage = math.floor((medium_correct / TOTAL_QUERIES) * 100)
    hard_correct_percentage = math.floor((hard_correct / TOTAL_QUERIES) * 100)

    return [easy_correct_percentage, medium_correct_percentage, hard_correct_percentage]

# print(compute_correct_percentage_results(FILE_PATH))


def compute_type_of_correct_results(json_file_path: str, result_type: str):
    json_result = {}

    with open(json_file_path) as f:
        json_result = json.load(f)

    # Initialise results
    completely_correct = 0
    partially_correct = 0
    barely_correct = 0

    for _, value in json_result.items():
        summary = value["summary"]

        # Calculate either for the response from GraphQL server or for the generated GraphQL query
        if result_type == "result":
            max_reward = summary["max_reward"]
        elif result_type == "query":
            max_reward = summary["max_query_reward"]
        else:
            raise ValueError("The value of 'type' parameter should either be 'result' or 'query'" )
        
        # Different ranges for completely, partially and barely correct
        if max_reward > 0 and max_reward < 0.55:
            barely_correct += 1
        elif max_reward > 0.55 and max_reward < 1:
            partially_correct += 1
        elif max_reward >= 1:
            completely_correct += 1

    # Calculate percentages
    completely_correct = math.floor((completely_correct / TOTAL_QUERIES) * 100)
    partially_correct = math.floor((partially_correct / TOTAL_QUERIES) * 100)
    barely_correct = math.floor((barely_correct / TOTAL_QUERIES) * 100)
    wrong = 100 - (completely_correct + partially_correct + barely_correct)

    return [completely_correct, partially_correct, barely_correct, wrong]


# print(compute_type_of_correct_results(FILE_PATH, "query"))
