from langchain import hub

from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent

from langchain.memory import ChatMessageHistory


from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun, human

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(
    model="mistral",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

python_repl = PythonREPL()
wikipedia = WikipediaAPIWrapper()
search = DuckDuckGoSearchRun()


tools = [
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="Useful for when you need to use python to answer a question. The input should be python code without markdown. Before sending code to the Python REPL tool, ensure any markdown formatting, such as triple backticks, is removed",
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to look up a topic, country or person on Wikipedia",
    ),
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input.",
    ),
]


prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(
    llm,
    tools,
    prompt,
)


# Function to remove markdown formatting from code
def remove_markdown_formatting(code_with_markdown):
    # Remove the triple backticks and any language specification, then strip leading/trailing whitespace
    clean_code = code_with_markdown.replace("```python", "").replace("```", "").strip()
    return clean_code


# Preprocess action input to remove markdown before sending it to the Python REPL tool
def preprocess_action_input(action_input):
    return remove_markdown_formatting(action_input)


agent_executor = AgentExecutor(
    agent=agent,
    tools=[
        Tool(
            name=tool.name,
            func=lambda input: (
                tool.func(preprocess_action_input(input))
                if tool.name == "python repl"
                else tool.func(input)
            ),
            description=tool.description,
        )
        for tool in tools
    ],
    verbose=False,
    max_iterations=5,
    handle_parsing_errors=True,
)


chat_history = ChatMessageHistory()


if __name__ == "__main__":
    print(prompt.template)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        chat_history.add_user_message(user_input)
        result = agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history.messages}
        )
        chat_history.add_ai_message(result["output"])
        print("\n")
