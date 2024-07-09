# Python built-in module
import time
import traceback
# Python installed module
import numpy as np
from sklearn.cluster import KMeans
from embed import Embed
from langchain.callbacks import get_openai_callback
# Python user defined module
from map_reduce_llama3 import MapReduce
from sentence_splitter import SentencizerSplitter
import warnings
warnings.filterwarnings("ignore")

class ClusterBasedSummary(object):
    '''This class implements the clustered based summarization'''
    
    def __init__(self, config_dict):
        self.embeddings = Embed().get()
        self.embedding_chunk_size = config_dict["embedding"]["chunk_size"]
        self.num_clusters = config_dict["cluster_summarization"]["num_clusters"]
        self.k = config_dict["cluster_summarization"]["num_closest_points_per_cluster"]
        self.config_dict = config_dict
        
        self.text_splitter = SentencizerSplitter(self.config_dict)
        self.map_reduce_summarizer = MapReduce(config_dict)
        
    def __call__(self, text_content):
        try:
            with get_openai_callback() as openai_cb:
                start_time = time.time()
                document_splits = self.text_splitter.create_documents(text_content)
                #total_splits = len(document_splits)
                vectors = self.embeddings.embed_documents(texts=[x.page_content for x in document_splits], chunk_size=self.embedding_chunk_size)
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(vectors)
                closest_indices = []
                for i in range(self.num_clusters):
                    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                    closest_index = np.argsort(distances)[:self.k]
                    closest_indices.extend(closest_index)
                    selected_indices = sorted(closest_indices)
                sorted_documents = [document_splits[idx].page_content for idx in selected_indices]
                mr_result_dict = self.map_reduce_summarizer("\n".join(sorted_documents), redirect="cluster_summary")
                end_time = time.time()
            t = {"summary": mr_result_dict["summary"],
                    "keywords": mr_result_dict["keywords"],
                    "metadata": { "total_time": round((end_time-start_time), 2)}}
            # print(type(t))
            return {"summary": mr_result_dict["summary"],
                    "keywords": mr_result_dict["keywords"],
                    "metadata": {"total_time": round((end_time-start_time), 2)}}
            
        except Exception as error:
            print("[ERROR] Some error happend in Map Reduce. Error:\n\n{}\n\n".format(error))
            traceback.print_exception(type(error), error, error.__traceback__)
            return