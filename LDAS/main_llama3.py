# Python built-in module
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import transformers
# import tokenizer
# Python installed module
import tiktoken
from dotenv import load_dotenv

# Python user defined module
from cod_llama3  import COD
from map_reduce_llama3 import MapReduce
from kmean import ClusterBasedSummary
import warnings
warnings.filterwarnings("ignore")
# Load the .env file where the openAI token is set

# env_found = load_dotenv("/workspace/.env")
# if not env_found:
#     print("[WARN] The .env file is not found. Please ensure `OPENAI_API_KEY` environment variable is set properly!")


# def validate_configs(config_dict):
#     validation_config_file_name = "validation_config.yaml"
#     # Load the validation config file
#     with open(validation_config_file_name, "r") as yaml_obj:
#         validation_config_dict = yaml.safe_load(yaml_obj)
    
#     input_file_extn_list = validation_config_dict["input_file_valid_extn"]
#     output_file_extn_list = validation_config_dict["output_file_valid_extn"]
#     max_num_closest_points_per_cluster = validation_config_dict["max_num_closest_points_per_cluster"]
#     max_medium_token_length = validation_config_dict["max_medium_token_length"]
#     llm_token_mapping = validation_config_dict["llm_token_mapping"]
    
#     success_status_message = "SUCCESS"
#     failure_status_message = "FAILED"
    
#     input_file_path = Path(config_dict["io_config"]["input_file"])
#     output_file_path = Path(config_dict["io_config"]["output_file"])
    
#     if input_file_path.suffix not in input_file_extn_list:
#         print("[ERROR] The input file extension should be one of the {}".format(",".join(input_file_extn_list)))
#         return failure_status_message
#     if output_file_path.suffix not in output_file_extn_list:
#         print("[ERROR] The output file extension should be one of the {}".format(",".join(output_file_extn_list)))
#         return failure_status_message
#     if config_dict["sentence_splitter"]["approx_total_doc_tokens"] + config_dict["sentence_splitter"]["tolerance_limit_tokens"] >= llm_token_mapping[config_dict["embedding"]["model_name"]]:
#         print("[ERROR] The sum of `approx_total_doc_tokens` and `tolerance_limit_tokens` in `sentence_splitter` config should not exceed the value of {} model's token limit!".format(config_dict["embedding"]["model_name"]))
#         return failure_status_message
#     if config_dict["summary_type_token_limit"]["short"] + config_dict["cod"]["max_tokens"] >= llm_token_mapping[config_dict["cod"]["model_name"]]:
#         print("[ERROR] The sum of `short` and `max_tokens` in `summary_type_token_limit` and `cod` config respectively should not exceed the value of {} model's token limit!".format(config_dict["cod"]["model_name"]))
#         return failure_status_message
#     if config_dict["cluster_summarization"]["num_closest_points_per_cluster"] > max_num_closest_points_per_cluster:
#         print("[ERROR] The `num_closest_points_per_cluster` in `cluster_summarization` config should not be more than {}".format(max_num_closest_points_per_cluster))
#         return failure_status_message
#     if config_dict["summary_type_token_limit"]["medium"] > max_medium_token_length:
#         print("[ERROR] The `medium` in `summary_type_token_limit` config should not be more than {} tokens.".format(max_medium_token_length))
#         return failure_status_message
#     return success_status_message
    
    
    

    
def generate_context(document):
    config_file_name = "config_llama3.yaml"
    unique_id = datetime.strftime(datetime.now(), "%d%m%y%H%M%S")
    
    # Load the config file
    with open(config_file_name, "r") as yaml_obj:
        config_dict = yaml.safe_load(yaml_obj)
        
    # Validate the configuarations
    # validation_status = validate_configs(config_dict)
    # if validation_status == "FAILED":
    #     return
        
    # Identify the summary type, short, medium or long form text summarization
    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

    text_content = document
    total_tokens = len(tiktoken_encoding.encode(text_content))
    total_words = int(total_tokens * 0.75)
    if total_tokens <= config_dict["summary_type_token_limit"]["short"]:
        chain_of_density_summarizer = COD(config_dict)
        result_dict = chain_of_density_summarizer(text_content)
    elif total_tokens <= config_dict["summary_type_token_limit"]["medium"]:
        map_reduce_summarizer = MapReduce(config_dict)
        result_dict = map_reduce_summarizer(text_content)
    else:
        cluster_summarizer = ClusterBasedSummary(config_dict)
        result_dict = cluster_summarizer(text_content)
    # print(type(result_dict))
    if type(result_dict).__name__ == "dict":
            return result_dict
        # print("[INFO] Summary output is successfully created in the output folder!")
        
