import gzip
import json
import os
import pandas as pd
from torch_geometric.data import Data, DataLoader
import torch
from sklearn.model_selection import train_test_split

def load_dataset(file_path: str) -> pd.DataFrame:
    data = []
    db = []    
    for file in file_path.split(' '):
        x = os.path.basename(os.path.dirname(file))
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            tmp = json.load(f)
            data = data + tmp
            db = db + [x]*len(tmp)
    data = pd.DataFrame(data)
    data = data.assign(db=db)
    return data

def create_dataset_from_dataframe(df, result=True):
    dataset = []
    for _, row in df.iterrows():
        edge_index = torch.tensor(row['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(row['edge_attr'], dtype=torch.float)
        num_nodes = row['num_nodes']

        y_raw = row.get('y', None)
        if result and y_raw is not None and isinstance(y_raw, list) and len(y_raw) > 0 and isinstance(y_raw[0], list):
            y = torch.tensor([y_raw[0][0]], dtype=torch.long)
        else:
            y = torch.tensor([0], dtype=torch.long)

        data = Data(
            x=torch.ones((num_nodes, 1)),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        data.x = torch.nan_to_num(data.x, nan=0.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)

        dataset.append(data)
    return dataset
