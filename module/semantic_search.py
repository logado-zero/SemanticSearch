from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import faiss
import numpy as np
import pandas as pd
import time
from .preprocess import preprocessing

from typing import List

class Semantic_Search:
    """ A semantic search engine bases on:
    The architecture: Sentence Transformer (SBERT)
        --> Turn data input to embedding
    The store and search method: FAISS (Facebook)
        --> Calculate the similarity between embedding

    """
    def __init__(self, model_name = "all-mpnet-base-v2",available_dataset = None ,device = None):
        """
        model_name :        name of sematic search model
        available_dataset:  path of file index FAISS dataset (Ex: dataset.index)
        device:             device running model _ "cpu", "cuda"
        """
        # Load model for embedding
        # device (str) = None: checks if a GPU can be used
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as ex:
            raise RuntimeError(f"Cannot load model from {model_name}: {ex}")
        

        # Load available dataset
        try:
            self.embed_index = faiss.read_index(available_dataset) if available_dataset is not None else None
        except Exception as ex:
            raise RuntimeError(f"Cannot read dataset FAISS index file from {available_dataset}: {ex}")
        


    def process_dataset(self, list_dataset: List, save_id_path = None):
        """
        Preprocessing and embedding dataset and saving faiss index

        list_dataset:   List of data contains texts want to embedding
        save_id_path:   Path to save file index FAISS
        """
        # Preprocessing
        list_dataset = [preprocessing(sen) for sen in list_dataset]

        # Embedding dataset
        encoded_data = self.model.encode(list_dataset)
        encoded_data = np.asarray(encoded_data.astype('float32'))
        shape_output = encoded_data.shape

        # Using FAISS
        # encoded with a n-dimensional vector
        self.embed_index = faiss.IndexIDMap(faiss.IndexFlatIP(shape_output[1])) 

        self.embed_index.add_with_ids(encoded_data, np.array(range(0, shape_output[0])))

        # stored to disk with attri_des.index name.
        save_id_path = "dataset/dataset.index" if save_id_path is None else save_id_path
        faiss.write_index(self.embed_index, save_id_path)


    def fetch_content_info(self, dataframe: pd.DataFrame, dataframe_idx: int, name_content: List,dist: int = None):
        """
        Extract information from dataframe

        dataframe:      Dataframe of dataset
        dataframe_idx:  Index of column dataframe want to extract
        name_content:   List of attribute dataframe want to extract
        dist:           The distance score want to show
        """
        info = dataframe.iloc[dataframe_idx]
        meta_dict = dict()
        for name in name_content:
            meta_dict[name] = info[name]
        if dist is not None:
            meta_dict["DISTANCE"] = dist
        return meta_dict
    
    def search(self, query: str, dataset: pd.DataFrame, name_content: List, top_k: int = 5 ):
        """
        Search base on query on dataset with model

        query:      query sentence for searching
        dataset:    Dataframe contains information results
        top_k:      number of the top result to show
        """
        t=time.time()
        # Preprocessing query input 
        query = preprocessing(query)

        # Embedding query input
        query_vector = self.model.encode([query])

        #Search from dataset index
        top_k = self.embed_index.search(query_vector, top_k)

        print('>>>> Results in Total Time: {}'.format(time.time()-t))
        # Collect top k results
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(top_k_ids)
        top_k_dist = top_k[0].tolist()[0]
        results =  [self.fetch_content_info(dataset, idx, name_content,dist) for dist, idx in zip(top_k_dist,top_k_ids)]
        return results




        
