from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 400, temperature=0.4)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke\n {joke}",
    input_variables=['joke']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, model, parser)
}

)

chain = RunnableSequence(joke_gen_chain, parallel_chain)

#result = chain.invoke({'topic' : 'President'})

#print(result)

chain.get_graph().print_ascii()