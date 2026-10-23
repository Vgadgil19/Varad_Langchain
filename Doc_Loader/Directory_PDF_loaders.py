from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-mini', max_tokens = 400, temperature=0.4)

loader = DirectoryLoader(
    path='DOcs',
    glob='**/*pdf',
    loader_cls=PyPDFLoader
)

docs = list(loader.lazy_load())
# print(len(docs))

#print(docs[23].page_content)

# if len(docs) > 23:
#     print(docs[2].page_content)
# else:
#     print("Doc is out of range!!")

# for documents in docs:
#     print(documents.metadata)

prompt1 = PromptTemplate(
    template="Answer the following question from given text\n {question} \n {text}",
    input_variables=['question', 'text']
)

prompt2 = PromptTemplate(
    template="Generate summary for the given {info}",
    input_variables=['info']
)

prompt3 = PromptTemplate(
    template="Generate 4 short Question Answers for the given {info}",
    input_variables=['info']
)


parser = StrOutputParser()

ans_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'info' : RunnablePassthrough(),
    'summary' : RunnableSequence(prompt2, model,parser),
    'qa' : RunnableSequence(prompt3, model, parser)
}  
)

chain = RunnableSequence(ans_chain, parallel_chain)

result = chain.invoke({'question' : 'What the given text tell us about ?', 'text' : docs[27].page_content })


print("\n" + "-------------------------")
print("Ans")
print("\n" + "-------------------------")
print(result['info'])
print("\n" + "-------------------------")
print("Summary")
print("\n" + "-------------------------")
print(result['summary'])
print("\n" + "-------------------------")
print("QA")
print("\n" + "-------------------------")
print(result['qa'])

#chain.get_graph().print_ascii()
