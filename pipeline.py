import torch
from torch.utils.data import Dataset,DataLoader
from datasets_files.dataset_explore import get_real_data
import torch.nn as nn
from data_tokenize import get_splits

#This line checks if you have a GPU else safely falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
print(f"Factory Power Source Selected: {device}")


class ReviewDataset(Dataset):
    def __init__(self,reviews,labels):
        #reviews:A list of strings
        #lables:A list of number(0 for negative,1 for positive)

        self.reviews = torch.tensor(reviews,dtype = torch.long)
        self.labels = torch.tensor(labels,dtype=torch.float32)
    
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self,idx):
        single_review = self.reviews[idx]
        single_label = self.labels[idx]

        return single_review,single_label

# Fetch tokenized splits from Rajeev's tokenize.py
train_x, train_y, val_x, val_y, test_x, test_y = get_splits(mock_data=True)

# Update your dataset instance to use Rajeev's train_x and train_y arrays
dataset = ReviewDataset(train_x, train_y)
dataLoader = DataLoader(dataset, batch_size=2, shuffle=True)

#real_reviews,real_labels = get_real_data()
#dataset = ReviewDataset(real_reviews,real_labels)
#dataLoader = DataLoader(dataset,batch_size = 2,shuffle = True)

# print("\n--- Testing Conveyor Belt (DataLoader) Batches ---")
# for batch_idx, (batch_reviews, batch_labels) in enumerate(dataLoader):
#     print(f"Box {batch_idx + 1} coming down the belt:")
#     print(f"  Processed Reviews: {batch_reviews}")
#     print(f"  Processed Labels:  {batch_labels}")

print("\n=============================================")
print("      WEEK 2: INITIALIZING EMBEDDING LAYER    ")
print("=============================================")


VOCAB_SIZE = 10000  
EMBEDDING_DIM = 100 

embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
print("Translation Machine successfully built and installed on the factory line.")

print("\n--- Testing 3D Shape Transformation on a Live Batch ---")

for batch_reviews,batch_labels in dataLoader:
    print(f"Step 1 (Partner's Output): Flat 2D barcode batch shape: {batch_reviews.shape}")
    
    embedded_output = embedding_layer(batch_reviews)
        
    print(f"Step 2 (Your Output): Machine translated it into a 3D block of coordinates: {embedded_output.shape}")
    print("   -> Dimensions mean: (Batch Size, Sequence Length[512], 100 Meaning Coordinates Per Word!)")
    
    break 
print("=============================================")