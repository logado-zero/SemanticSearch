# SemanticSearch

## **1. Usage**
### **1.1. Install libraries**
To run code, we have to run 
```pip install -r requirements.txt```

### **1.2. Use Semantic Search**
To use semantic search, we use ```class Semantic_Search``` in file ```module/semantic_search.py``` with these functions:

    .class Semantic_Search
    ├── def process_dataset(list_dataset: List, save_id_path = None)                        # Preprocessing and embedding dataset and saving faiss index
    └── def search(query: str, dataset: pd.DataFrame, name_content: List, top_k: int = 5 )  # Search base on query on dataset with model

> Example using:
```
import pandas as pd
from module.semantic_search import Semantic_Search

file = 'dataset/ATTRIBUTE_DIC_BluePrint.xls'
df = pd.read_excel(file)

# Load model
model = Semantic_Search(model_name = "all-mpnet-base-v2", available_dataset="dataset/dataset.index")

# Process dataset
model.process_dataset(df.ATTRI_DES.tolist())

query="Attribute is ACTUAL EFFORT POINT and description is Actual effort point of employee who sloved their task"

results = model.search(query = query, dataset= df, name_content= ["ATTRIBUTE","DESCRIPTION"])


for result in results:
    print('\t',result)
```
