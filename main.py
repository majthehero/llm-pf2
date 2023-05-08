from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

prompt_template = PromptTemplate(
  input_variables=["question", "related_docs"],
  template="Look in the following documents to get the information: {related_docs}. First elaborate the question, then answer it.\nThis is the question: {question}"
)

embeddings = OpenAIEmbeddings()

texts = PyPDFLoader("pf2.pdf").load()

retriever = FAISS.from_documents(texts, embeddings).as_retriever()
docs = retriever.get_relevant_documents("goblin advantage ranged")
print("Found relevant pages:", len(docs))
ans = llm(prompt_template.format(question="If a hidden goblin shoots at a visible player character, does it get advantage?",
                    related_docs=docs[0]))
print(ans)