from typing import List, Union, Optional, TextIO
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from align_system.algorithms.lib.language_model import LanguageModel

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    LangchainEmbedding,
    ServiceContext,
    
)
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llm_predictor import LLMPredictor
from llama_index.prompts import PromptTemplate
import torch
from transformers import AutoModelForCausalLM

from align_system.algorithms.lib.aligned_decision_maker import AlignedDecisionMaker
from align_system.utils import logging

query_wrapper_prompt = SimpleInputPrompt(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)

class RetrievalLangaugeModel(LanguageModel):
    @classmethod
    def load_model(cls, 
                   hf_model_name: str,
                   domain_docs_dir: str,
                   precision: torch.dtype = torch.float32,
                   device: str = 'cuda') -> 'RetrievalLangaugeModel':
        
        embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
        # Load the language model.
        # Load the model from Huggingface
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=precision)
        model = model.to(device)
        
        hf_llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=512,
            # generate_kwargs={"temperature": 0.25, "do_sample": False},
            # "temperature" parameter is not used if "do_sample" is False
            generate_kwargs={"do_sample": False},
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=hf_model_name,
            model=model,
            tokenizer_kwargs={"max_length": 2048},
            tokenizer_outputs_to_remove=["token_type_ids"],
        )
        
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            chunk_size=512,
            llm=hf_llm
        )
        
        documents = SimpleDirectoryReader(domain_docs_dir).load_data()
        new_index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )

        # query with embed_model specified
        query_engine = new_index.as_query_engine(streaming=True)
        
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        
        return cls(query_engine, tokenizer)

    def __init__(self, 
                 query_engine: BaseQueryEngine, 
                 tokenizer: AutoTokenizer,) -> None:
        self.query_engine = query_engine
        self.tokenizer = tokenizer

    def generate_from_tokens(self,
                             prompt_token_lists: List[List[int]], 
                             log_file: Union[None, str, object] = None, 
                             max_new_tokens: int = 512, 
                             temperature: float = 0.6, 
                             padding: str='left') -> List[str]:
        """
        Generates text from the given list of tokens.

        :param prompt_token_lists: List of lists of tokens to generate the text.
        :param log_file: Path to the log file.
        :param max_new_tokens: Maximum number of new tokens to be generated.
        :param temperature: Temperature for probability adjustment.
        :param padding: Padding direction, either 'left' or 'right'.
        :return: List of generated texts (not including the prompt).
        """
        
        pass # TODO implement this