import json

file = "All_Beauty.jsonl"
reviews = []
with open(file, 'r') as fp:
    for line in fp:
        reviews.append(json.loads(line.strip()))
print(reviews[0])