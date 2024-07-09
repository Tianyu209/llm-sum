# Python built-in module
# from langchain_core.runnables import RunnablePassthrough
from time import time
import traceback
# Python installed module
# from langchain_community.llms import Ollama

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

from langchain.schema import HumanMessage
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain_core.documents import Document

# Python user defined module
import prompts
from Groq_models import KWGroq, CODGroq
from cod_llama3 import COD
from sentence_splitter import SentencizerSplitter
import warnings
warnings.filterwarnings("ignore")

class MapReduce(object):
    '''This class implements the Map Reduce summarization'''
    
    def __init__(self, config_dict):
        self.chat_4_llm = CODGroq(config_dict).get()
        
        self.chat_turbo_llm = KWGroq(config_dict).get()
        # self.chat_4_llm = Ollama(model=config_dict["cod"]["model_name"],
        #                             num_gpu=1,
        #                              num_predict = config_dict["cod"]["max_tokens"],
        #                              repeat_penalty = config_dict["cod"]["presence_penalty"],
        #                            temperature=config_dict["cod"]["temperature"],
        #                            top_p=config_dict["cod"]["top_p"])
        # self.chat_turbo_llm = Ollama(model=config_dict["kw_extract"]["model_name"],
        #                             num_gpu=1,
        #                              num_predict=config_dict["kw_extract"]["max_tokens"],
        #                              repeat_penalty=config_dict["kw_extract"]["presence_penalty"],
        #                            temperature=config_dict["kw_extract"]["temperature"],
        #                             top_p=config_dict["kw_extract"]["top_p"])
        
        #kw_extraction_prompt_template = PromptTemplate(input_variables=["text_chunk"], template=prompts.KW_EXTRACT_SYSTEM_PROMPT)
        #kw_extraction_chain = LLMChain(llm=self.chat_turbo_llm, prompt=kw_extraction_prompt_template, output_key="key_words")
        #summary_prompt_template = PromptTemplate(input_variables=["text_chunk", "key_words"], template=prompts.SEQUENCIAL_SUMMARY_PROMPT)

        summary_prompt_template = PromptTemplate(input_variables=["text_chunk"], template=prompts.SEQUENCIAL_SUMMARY_PROMPT2)
        # summary_chain = summary_prompt_template|self.chat_turbo_llm
        summary_chain = LLMChain(llm=self.chat_turbo_llm, prompt=summary_prompt_template, output_key="summary")
        self.extractive_summary_chain = SequentialChain(
                                                            chains=[summary_chain],
                                                            input_variables=["text_chunk"],
                                                            output_variables=["summary"],
                                                            verbose=False
                                                       )
        
        reduce_prompt = PromptTemplate.from_template(prompts.REDUCE_PROMPT)
        reduce_chain = LLMChain(llm=self.chat_4_llm, prompt=reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=config_dict["cod"]["max_tokens"],
        )
        self.config_dict = config_dict
        self.text_splitter = SentencizerSplitter(self.config_dict)
        self.chain_of_density_summarizer = COD(self.config_dict)
        
    def __call__(self, text_content, redirect=None):
        try:
            start_time = time()
            summaries_list = list()
            do_cod = False
            document_splits = self.text_splitter.create_documents(text_content)
            total_splits = len(document_splits) 
            # print(total_splits)
            t1 = time()
            # with get_openai_callback() as openai_cb:
            for idx, doc in enumerate(document_splits, start=1):
                extractive_summary_result = self.extractive_summary_chain({"text_chunk": doc.page_content})
                summaries_list.append(extractive_summary_result["summary"])
            # print("\n"+str(summaries_list)+"\n")

            final_summary = self.reduce_documents_chain.invoke([Document(page_content=chunk) for chunk in summaries_list])
            kw_extract_messages = [HumanMessage(content=prompts.KW_EXTRACT_SYSTEM_PROMPT.format(text_chunk=final_summary))]
            kw_response = self.chat_turbo_llm(kw_extract_messages)
            kw_output = kw_response.content.split(", ")
            t2 = time()
            # print("time for es llm is "+str(t2-t1))
            if (redirect == "cluster_summary") and (self.config_dict["cluster_summarization"]["final_dense"] == True):
                do_cod = True
            elif (redirect == "cluster_summary") and (self.config_dict["cluster_summarization"]["final_dense"] == False):
                do_cod = False
            elif (redirect is None) and (self.config_dict["cluster_summarization"]["final_dense"] == True):
                do_cod = True
            elif (redirect is None) and (self.config_dict["cluster_summarization"]["final_dense"] == False):
                do_cod = False
                
            if do_cod:
                cod_result_dict = self.chain_of_density_summarizer(final_summary)
                end_time = time()
                return {"summary": cod_result_dict["summary"],
                        "keywords": cod_result_dict["keywords"],
                        "metadata": {"total_time": round((end_time-start_time), 2)}}
            
            end_time = time()
            return {"summary": final_summary,
                    "keywords": kw_output,
                    "metadata": {"total_time": round((end_time-start_time), 2)}}
        except Exception as error:
           # print("[ERROR] Some error happend in Map Reduce. Error:\n\n{}\n\n".format(error))
            traceback.print_exception(type(error), error, error.__traceback__)
            return