from langchain_openai import OpenAIEmbeddings
import os 
import warnings
warnings.filterwarnings("ignore")

class Embed(object):
    def __init__(self):
            self.embeddings = OpenAIEmbeddings(  
model="text-embedding-3-small" ,  
openai_api_key=os.environ.get("OPENAI_API_KEY")  
)  
    def get(self):
        return self.embeddings
        