from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)                        
print(dataset["train"][0])            
print(dataset["train"].features)