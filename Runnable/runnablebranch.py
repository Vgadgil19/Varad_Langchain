from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence, RunnableLambda, RunnableBranch

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 400, temperature=0.4)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a detailed report in 400 words for the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following {text}",
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

conditional_chain = RunnableBranch(
    (lambda x : (len(x.split())) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen_chain, conditional_chain)

result = chain.invoke({'topic' : 'Ramayan'})

print(result)