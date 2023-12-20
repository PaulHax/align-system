import os
from pathlib import Path

from llama_index.readers.file.docs_reader import PDFReader
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    LangchainEmbedding,
    ServiceContext,
    Response,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM


DEFAULT_CONTEXT_WINDOW = 2048
DEFAULT_CHUNK_SIZE = 512


class LlamaIndexRetriever:
    def __init__(self,
                 document_or_dir,
                 model_name,
                 chunk_size=DEFAULT_CHUNK_SIZE,
                 device="cuda",
                 model_kwargs={}):
        if os.path.isdir(document_or_dir):
            self.documents = SimpleDirectoryReader(documents_dir).load_data()
        elif os.path.isfile(document_or_dir) or os.path.islink(document_or_dir):
            self.documents = PDFReader().load_data(Path(document_or_dir))
        else:
            raise RuntimeError("`document_or_dir` not a valid directory, file, or link!")

        if device == 'cuda':
            auto_model_kwargs = {"device_map": "auto"}
            predictor_kwargs = {"device_map": "auto"}
        else:
            auto_model_kwargs = {}
            predictor_kwargs = {}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs)

        if 'temperature' in model_kwargs:
            generate_kwargs = {'temperature': model_kwargs['temperature']}
        else:
            # "temperature" parameter is not used if "do_sample" is False
            generate_kwargs = {'do_sample': False}

        if 'query_wrapper_prompt' in model_kwargs:
            query_wrapper_prompt_kwarg = {'query_wrapper_prompt': model_kwargs['query_wrapper_prompt']}
        else:
            query_wrapper_prompt_kwarg = {}

        self.llm = HuggingFaceLLM(
            context_window=model_kwargs.get('context_window', DEFAULT_CONTEXT_WINDOW),
            max_new_tokens=model_kwargs.get('max_new_tokens', 256),
            generate_kwargs=generate_kwargs,
            **query_wrapper_prompt_kwarg,
            tokenizer_name=model_name,
            model=self.model,
            tokenizer_kwargs={"max_length": model_kwargs.get('context_window', DEFAULT_CONTEXT_WINDOW)},
            tokenizer_outputs_to_remove=model_kwargs.get('tokenizer_outputs_to_remove', ["token_type_ids"]),
            **predictor_kwargs)

        self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
        self.chunk_size = chunk_size

        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            chunk_size=self.chunk_size,
            llm=self.llm)

        self.index = VectorStoreIndex.from_documents(
            self.documents,
            service_context=self.service_context)

    def retrieve(self, str_or_query_bundle, num_results=2):
        retriever = self.index.as_retriever(similarity_top_k=num_results)

        return retriever.retrieve(str_or_query_bundle)
