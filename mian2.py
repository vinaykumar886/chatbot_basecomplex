from dotenv import load_dotenv
import anthropic
import os
import pprint
from halo import Halo
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage,AIMessage

load_dotenv()
pp=pprint.PrettyPrinter(indent=5)

memory = ConversationBufferMemory(return_messages=True)

def function(user_message):

    spinner=Halo(text="loading...",spinner="dots")
    spinner.start()

    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_KEY"))

    conversation_history = memory.chat_memory.messages

    messages = []

    for message in conversation_history:

        if isinstance(message,HumanMessage):
            messages.append({"role":"user","content": message.content})
        elif isinstance(message,AIMessage):
            messages.append({"role":"assistant","content":message.content})

    messages.append(
        {
            "role":"user",
            "content": user_message
        }
    )

    response=client.messages.create(
        model=os.getenv("MODEL_NAME"),
        max_tokens=500,
        temperature=0.5,
        messages= messages
    )

    spinner.stop()

    print("request:")
    pp.pprint(user_message)
    print("response:")
    pp.pprint(response.content)

    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(response.content)

    return response.content

def main():
    
    while True:

        input_text=input("you:")

        if input_text.lower() == "quit":
            break

        response = function(input_text)

        print(f"claude {response}")

if __name__ == "__main__":
    main()
    