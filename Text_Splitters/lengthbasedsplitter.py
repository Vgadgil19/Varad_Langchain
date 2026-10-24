from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('AMBA.pdf')
# docs = list(loader.load())
# print(len(docs))
docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    separator=''
)

# result = list(splitter.split_documents(docs))
# print(len(result))

result = splitter.split_documents(docs)
print(result[127].page_content)