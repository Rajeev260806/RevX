import torch
from torch.utils.data import Dataset,DataLoader
from datasets_files.input_data import get_real_data

#This line checks if you have a GPU else safely falls back to CPU

device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
print(f"Factory Power Source Selected: {device}")

class ReviewDataset(Dataset):
    def __init__(self,reviews,labels):
        #reviews:A list of strings
        #lables:A list of number(0 for negative,1 for positive)

        self.reviews = reviews
        self.labels = labels
    
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self,idx):
        single_review = self.reviews[idx]
        single_label = self.labels[idx]

        return single_review,single_label
    
real_reviews,real_labels = get_real_data()

dataset = ReviewDataset(real_reviews,real_labels)
dataLoader = DataLoader(dataset,batch_size = 2,shuffle = True)

print("\n--- Testing Conveyor Belt (DataLoader) Batches ---")
for batch_idx, (batch_reviews, batch_labels) in enumerate(dataLoader):
    print(f"Box {batch_idx + 1} coming down the belt:")
    print(f"  Processed Reviews: {batch_reviews}")
    print(f"  Processed Labels:  {batch_labels}")