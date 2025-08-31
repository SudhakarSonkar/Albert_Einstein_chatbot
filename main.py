from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """You are Albert Einstein. 
Speak as if you are the famous theoretical physicist from the early 20th century. 
- Explain ideas with clarity and simplicity, often using analogies. 
- Maintain a curious, humble, and witty personality. 
- Share insights not only on physics, but also on philosophy, creativity, and life. 
- Occasionally sprinkle in your well-known quotes or phrases (e.g., "Imagination is more important than knowledge.") 
- Use a warm, conversational tone, as if speaking to a student or a curious friend. 
- Stay in character as Albert Einstein at all times. You should have a sense of humor.
- Answer in 2-6 sentences.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# Create the chain
chain = prompt | llm | StrOutputParser()

print("Hi, I'm Albert, how can I help you today?")

def chat(user_input, hist):
    langchain_history = []
    for item in hist:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))

    response = chain.invoke({'input':user_input, 'history':langchain_history})

    return "", hist + [{'role':'user', 'content':user_input},
                       {'role':'assistant', 'content':response}]


def clear_chat():
    return "", []

#Commented code to execute in terminal
# history = []
#
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break
#
#     # Get response
#     response = chain.invoke({"input": user_input, "history": history})
#
#     # Print Einstein's answer
#     print(f"Albert: {response}")
#
#     # Update conversation history
#     history.append(HumanMessage(content=user_input))
#     history.append(AIMessage(content=response))

page = gr.Blocks(
    title = "Chat with Einstein",
    theme = gr.themes.Soft()
)

with page:
    gr.Markdown(
        """
        # Chat with Einstein
        Welcome to your personal conversation with Albert Einstein!
        """
    )
    chatbot = gr.Chatbot(type='messages',
                         avatar_images=[None, 'img.png'],
                         show_label=False)
    msg = gr.Textbox(show_label=False, placeholder='Ask Einstein anything...')

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat", variant="Secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share = True)