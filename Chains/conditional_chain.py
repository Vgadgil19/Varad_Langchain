from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field,BaseModel
from typing import Literal

load_dotenv()

class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description='Analyze the sentiment of the given feedback')

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)
model = ChatOpenAI(model='gpt-4.1-mini', temperature=0.4, max_tokens = 200)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables= ['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response for this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response for this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x : x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "couldn't recognise the sentiment")
)

chain = classifier_chain | branch_chain
print(chain.invoke({'feedback': 'This phone is a waste of maney!'}))

chain.get_graph().print_ascii()
