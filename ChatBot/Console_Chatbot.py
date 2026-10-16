from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
#model = 'gpt-4.1-mini', temperature = 0.1, max_tokens = 50
model  = ChatOpenAI(model = 'gpt-4.1-mini', temperature = 0.1, max_tokens = 50)
chat_history = []

while True:
    user_input = input("You : ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        print("Exiting from chatbot!")
        break   
    response = model.invoke(chat_history)
    chat_history.append(response)
    print("AI : ", response.content)
print(chat_history)
