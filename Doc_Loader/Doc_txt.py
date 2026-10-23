from langchain_classic.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 400, temperature=0.4)

loader = TextLoader('Cricket.txt', encoding='utf-8')
docs = loader.load()
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})
print(result)

chain.get_graph().print_ascii()