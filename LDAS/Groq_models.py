from langchain_groq import ChatGroq
import os
from langchain.schema import HumanMessage, SystemMessage
import warnings
warnings.filterwarnings("ignore")
#import prompts
    # Inistiating a new instance with given config and prompts
groq_api_key = os.environ.get("GROQ_API_KEY")  
class KWGroq(object):
    def __init__(self, config_dict,):
        self.llm = ChatGroq(model=config_dict["kw_extract"]["model_name"],
                       temperature=config_dict["kw_extract"]["temperature"],
                       max_tokens=config_dict["kw_extract"]["max_tokens"],
                       model_kwargs={"top_p": config_dict["kw_extract"]["top_p"],
                                     "presence_penalty": config_dict["kw_extract"]["presence_penalty"],
                                     "frequency_penalty": config_dict["kw_extract"]["frequency_penalty"]},
                      api_key = groq_api_key)
        
    @classmethod 
    def kw_extract_messages(text_content):
        return  [HumanMessage(content=prompts.KW_EXTRACT_SYSTEM_PROMPT.format(text_chunk=text_content))]
    def get(self):
        return self.llm

    def __call__(self, text_content):
        return self.llm(self.kw_extract_messages(text_content))
        
class CODGroq(object):
    def __init__(self, config_dict):
        self.llm = ChatGroq(model=config_dict["cod"]["model_name"],
                       temperature=config_dict["cod"]["temperature"],
                       max_tokens=config_dict["cod"]["max_tokens"],
                       model_kwargs={"top_p": config_dict["cod"]["top_p"],
                                     "presence_penalty": config_dict["cod"]["presence_penalty"],
                                     "frequency_penalty": config_dict["cod"]["frequency_penalty"]},
                      api_key = groq_api_key)

    @classmethod 
    def cod_messages(text_content):
        return [SystemMessage(content=prompts.COD_SYSTEM_PROMPT),
                                HumanMessage(content="Here is the input text for you to summarize using the 'Missing_Entities' and 'Denser_Summary' approach:\n\n{}".format(text_content))]
    def get(self):
        return self.llm
        
    def __call__(self, text_content):
        return self.llm(self.cod_messages(text_content))
        
class NormGroq(object):
    def __init__(self):
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            api_key = groq_api_key
        )
    def get(self):
        return self.llm
    @classmethod
    def __call__(self,prompt):
        return self.llm.invoke(prompt)