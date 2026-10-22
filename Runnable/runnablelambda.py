from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence, RunnableLambda

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

#for counting words in joke--------->
paralle_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explain' : RunnableSequence(prompt2, model, parser), 
    'word_count' :  RunnableLambda(lambda x :len(x.split()))
})

#for counting words in explaination--------->
# paralle_chain = RunnableParallel({
#     'joke' : RunnablePassthrough(),
#     'explain' : RunnableSequence(prompt2, model, parser, RunnableLambda(lambda x :{'explain' : x, 'word_count' : len(x.split())} ))
# })


chain = RunnableSequence(joke_gen_chain, paralle_chain)

result = chain.invoke({'topic' : 'Lion'})
print(result)

chain.get_graph().print_ascii()