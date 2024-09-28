from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_name = "KBTG-Labs/THaLLE-0.1-7B-fa"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./model-cache')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./model-cache', device_map=device, torch_dtype=torch.bfloat16)

class Qwen(LLM, ABC):
     max_token: int = 10000
     temperature: float = 0.01
     top_p: float = 0.9
     history_len: int = 3

     def __init__(self):
         super().__init__()

     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
         messages = [
             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
             {"role": "user", "content": prompt}
         ]
         text = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
         generated_ids = model.generate(
             **model_inputs,
             max_new_tokens=4096
         )
         generated_ids = [
             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
         ]

         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         return response

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}


CSV_File = '../Data/Article.csv'
loader = CSVLoader(CSV_File)
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
chunks = splitter.split_documents(data)
# print(len(chunks))
# print(len(chunks[1].page_content))
# print(chunks[1].page_content)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-xlm-r-multilingual-v1')
vectorstore = FAISS.from_documents(chunks,embeddings)

# retriever = vectorstore.as_retriever()
# ret = retriever.invoke("ขอดู Data ของหุ้น PTT ปี 1Q67 ของหุ้น PTT")
# print(ret)

prompt_template = """
You are an assistant that provides answers to questions based on
a given context. 

Answer the question based on the context. If you can't answer the
question, reply "I don't know".

Be as concise as possible and go straight to the point.

Context: {context}

Question: {question}
"""

# prompt = PromptTemplate.from_template(template)

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# print(prompt.format(context ="Here  is come context",question="Here is a question"))

llm = Qwen()
chain_type_kwargs = {"prompt": prompt,"document_variable_name": "context"}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs=chain_type_kwargs)

query = "เขียนบทความเกี่ยวกับผลประกอบการ 2Q67 ของ PTTEP และแนวโน้มในไตรมาส 3 รวมถึงปัจจัยเสี่ยงที่เกี่ยวข้อง คำแนะนำด้านการลงทุน และเป้าหมายราคา ขอบทความเป็นภาษาไทย"
print(qa.invoke(query))

