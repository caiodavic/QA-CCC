import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from rouge import Rouge

class SearchEngine:
    def __init__(self, index_path:str =None, context_path:str =None, model_path:str = None, threshold:float = 46.0):
            self.index = faiss.read_index(index_path) if index_path is not None else faiss.read_index('../database_context.index')  
            self.df = pd.read_csv(context_path) if context_path is not None else pd.read_csv('../database_context.csv')
            self.model = SentenceTransformer(model_path) if model_path is not None else SentenceTransformer('rufimelo/Legal-BERTimbau-sts-base')
            self.threshold = threshold
            self.rouge = Rouge()
    
    def fetch_context(self,df_idx:list) -> dict:
        info = self.df.iloc[df_idx]

        meta_dict = dict()

        meta_dict['id'] = info['id']
        meta_dict['question'] = info['question']
        meta_dict['answer'] = info['answer']
        meta_dict['category'] = info['category']
        return meta_dict
    
    def search(self,query:str, top_k:int=5) -> tuple[list,list]:
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        top_k_ids = top_k[1].tolist()[0]    
        uniq_top_k, idxs_top_k = np.unique(top_k_ids, return_index=True)
        top_k_ids = uniq_top_k[np.argsort(idxs_top_k)]
        results =  [self.fetch_context(idx) for idx in top_k_ids]
        return results, top_k[0]
    
    def first_result_func(self, query:str, predicts:list, sims:list)-> tuple[dict,float]:
        best = predicts[0]
        best_number = sims[0]
        for idx, i in enumerate(predicts):
            if sims[idx] < self.threshold:                      
                rouges = self.rouge.get_scores(query, i['question'])
                if rouges[0]['rouge-1']['f'] > best_number:
                    best_number = rouges[0]['rouge-1']['f']
                    best = i 
        return best, best_number

    def search_context(self,query:str,top_k:int=5)-> dict:
        results, sims = self.search(query, top_k=top_k)
        first_result, best_sim = self.first_result_func(query, results,sims[0])
        if best_sim < self.threshold:
            return first_result
        else:
            return None