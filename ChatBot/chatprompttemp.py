from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4.1-mini', temperature = 0.1, max_tokens = 50)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} assistant.'),
    ('human', 'Give me more information about {topic}.')
])

prompt = chat_template.invoke({'domain':'technology', 'topic' : 'AI'})
print(prompt)