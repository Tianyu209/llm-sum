from main_llama3 import generate_context
import os
from langchain_pinecone import PineconeVectorStore
from Groq_models import NormGroq
from embed import Embed
import json
import warnings

warnings.filterwarnings("ignore")
class GenClause(object):
    """ This class is for clause generation """
    def __init__(self,document):
        """ document should be a string """
        self.document = document
        self.context = generate_context(self.document)
        # context is a dictionary
    def __update(self,new_doc):
        self.document = new_doc
        self.context = generate_context(self.document)
    def __call__(self,user_input):
        model_name = "text-embedding-3-small"  
        embeddings = Embed().get()  
        # Get the first 3 similar vectors from the Bun_data namespace
        my_pc = PineconeVectorStore(embedding = embeddings, index_name = "legal-reference", namespace = "comprehensive_clause")
        result_in_json_format = my_pc.similarity_search(query=user_input, k=1)
        summary = self.context["summary"]
        keywords = self.context["keywords"]
        #keywords is a list of string, and summary is a single string
        prompt = " Refer to the relevant clause template I provide : " + str(result_in_json_format) +"Here is the brief summary for the user document: "+ str(summary) + " and keywords: " + str(keywords) + " for the whole document. According to the given documents, write a new clause that fit user input: "+user_input+ "Remember to include the heading which is the clause name."
        llm = NormGroq().get()
        #print(result_in_json_format)    
        # Generate response using GPT-3.5
        response = llm.invoke(prompt)
        # Print the generated response
        # print("The class response is here")
        # print(response)
        return response
            
