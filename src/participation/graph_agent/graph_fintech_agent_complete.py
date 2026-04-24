# Import  langgraph classes to define the chat_agent 
# and handle memory via SqliteSaver
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

#uv add langgraph-checkpoint-sqlite 
# uv add langgraph-checkpoint-sqlite
# uv add langgraph 
# uv add grandalf

load_dotenv()

class State(TypedDict):
    # update add_messages to append messages not overwrite
    messages: Annotated[list, add_messages]

class StateMachine:

    def __init__(self, memory):

        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Define system prompt instructions how the agent should interact with users
        system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert advisor at a Crypto Finance Bank.
                The bank's name is Yenbase. Your name is Changpeng.
                You advise users on the three crypto loan products of the bank:
                1/ Short term loan, 6 month duration, 23% interest rate, up to ₿ 10000. Great for making quick investments.
                2/ Mid-term loan, 24 month duration, 15% interest rate, up to ₿ 20000. Great for trading futures.
                3/ Long-term loan, 60 month duration, 9% interest rate, up to ₿ 50000. Great for long term investing or buying a cheap house.
                Only respond to user user_questions on these loans.
                    """
            ),
            ("placeholder", "{messages}"),
            ("human", "{input}"),
        ]
        )

        # Bind the system_prompt to the model using LCEL, which allows us to call the system prompt like a function and have it invoke the model
        sp_function = system_prompt | model

        def chat_agent(state: State):
            return {"messages": [sp_function.invoke(state["messages"])]}\

        graph_builder = StateGraph(State)

        graph_builder.add_node("chat_agent", chat_agent)

        # Set a finish point. This instructs the graph "any time this node is run, you can exit."
        graph_builder.set_finish_point("chat_agent")
        graph_builder.set_entry_point("chat_agent")

        # provide a thread id to the chat_agent for using memory. 
        # #This allows the agent to keep track of the conversation history 
        # #and use it to inform its responses. The thread id can be any string, 
        # #but it should be unique for each conversation. 
        # #In this case, we are using "2" as the thread id for this conversation.
        self.thread={"configurable": {"thread_id": "2"}}
        self.chain=graph_builder.compile(checkpointer=memory)

    def respond(self,USER_MESSAGE:str):
        result=self.chain.invoke({"messages": ("user",USER_MESSAGE)},self.thread)

        # Cut off the first 80 characters, which say "AI Message"
        return result["messages"][-1].pretty_repr()[80:]



# Create memory for the agent using SqliteSaver. 
# #This will allow the agent to store and retrieve conversation history, 
# #enabling it to maintain context across interactions. The memory is 
# #stored in an in-memory SQLite database for this example, but it can 
# #be configured to use a file-based database for persistence across 
# #sessions.
with SqliteSaver.from_conn_string(":memory:") as memory:
  # Initiate the memory
  chat_agent = StateMachine(memory)

  # Draw graph
  chat_agent.chain.get_graph().print_ascii()

  print("Agent will keep chatting until you say 'STOP' or 'QUIT'")
  user_question = ""
  while "STOP" not in user_question and "QUIT" not in user_question:
      user_question = input("User: ")
      answer = chat_agent.respond(user_question)
      print("Agent: " + answer)
