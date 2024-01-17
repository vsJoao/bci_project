import json
import numpy as np

filepath = 'C:/Users/victo/Desktop/file_bcicompetition.txt'

with open(filepath, 'r') as file:
    results = file.read()

results = json.loads(results)["exDict"]
print(results)

for method in results.keys():
    for user in results[method].keys():
        kappa = results[method][user]["kappa"]
        acc = results[method][user]["accuracy"]
        print(f"{method},{user},{kappa},{acc}")