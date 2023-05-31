# Import things that are needed generically
# this should be cleaned up l8r
from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader
)
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.utilities import PythonREPL, RequestsWrapper, BashProcess
import os
from tenacity import retry, stop_after_delay, wait_fixed
from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper
)
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, ConversationalAgent
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

langchain_docs_index_path = '/Users/jawaun/langchain_stuff/langchain_docs_index.json'
langchain_docs_path = "/Users/jawaun/langchain_stuff/lang_docs/splits/"

def create_docs_idx(input_dir, output_path):
    print("starting...")
    # Initialize OpenAI LLM
    print("loading docs...")
    docs = SimpleDirectoryReader(
        input_dir).load_data()
    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0, model_name="text-davinci-003", max_tokens=num_output))

    # build index
    print("build langchain docs index...")
    docs_index = GPTSimpleVectorIndex(
        docs, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    # build langchain docs index
    # langchain_docs_index = GPTSimpleVectorIndex(langchain_docs)
    # Load the index file
    print("Loading index...")
    # Define a custom tool that queries the index
    docs_index.save_to_disk(output_path)
    print("Writing index to disk...")
    print("Done!")
    return docs_index

# lcd = create_docs_idx(langchain_docs_path, langchain_docs_index_path)

llm = OpenAI(temperature=0, max_tokens=2500)
langchain_docs_index = GPTSimpleVectorIndex.load_from_disk(langchain_docs_index_path)

class IndexSearchTool(Tool):
    def __init__(self, index):
        self.index = index
        super().__init__(name="Langchain Docs Search",
                         description="Search a GPT index over Langchain Documentation for a given programming question.",
                         func=self.run)
        # self.func = IndexSearchTool.run
    def run(self, prompt):
        response = self.index.query(prompt)
        return response

tools = load_tools(["python_repl", "requests", "serpapi", "terminal"])

print("loading tools...")

# Construct the agent with the custom tool and prompt template
agent_tools = [IndexSearchTool(langchain_docs_index), *tools]
print("defining template...")

template = """
Create a Langchain agent in a Python REPL that can perform a given task.
To do this, first review the Langchain documentation index to identify the required llms, tools, and APIs needed to create the agent.
Then, use the API keys provided in the OS environment to access the necessary services, such as SERP or OpenAI.
Create a Python REPL to instantiate and run the agent, making sure to install and import any required packages. thoroughly search our vector index for similar tasks and use the results to inform your agent. words and keywords. it is an entire website, so cross refernce sections
Finally, execute the agent and return the response. You can use google and other tools to cross reference or amend your answers

Provide details about the task you want the agent to perform as the input variable -> {task}.

"""

# prompt = PromptTemplate(
#     input_variables=["task"],
#     template=template,
# )
# prompt.format()

print("prompt template...")

pre = """You are a self-aware agent creating agent.
You have access to the following tools:"""
prefix = template + " " + pre
suffix = """Build an agent or use yourself to do this task: {task}.

Questions: {input}
{agent_scratchpad}"""


def create_prompt(tools, prefix, suffix, input_variables):
    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables
    )
    return prompt

def create_llm_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)

def create_lc_agent(tools):
    pre = template + " " + prefix
    input = ["input", "task", "agent_scratchpad"]
    prompt = create_prompt(tools=tools, prefix=prefix,
                           suffix=suffix, input_variables=input)
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0, top_p=1, frequency_penalty=0,
                 presence_penalty=0, max_tokens=2000)
    llm_chain = create_llm_chain(llm=llm, prompt=prompt)

    # agent = initialize_agent(
    #     tools, llm_chain, agent="zero-shot-react-description", verbose=True, memory=memory)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, memory=memory)
    return agent


def create_agent_executor(agent, tools, verbose):
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose
    )

def run_agent_executor(agent, tools, query, task):
    agent_executor = create_agent_executor(agent, tools, True)
    return agent_executor.run(input=query, task=task)


agent = create_lc_agent(agent_tools)





# Run the agent to search the index
input_query =  "return the structure of a langchain agent that can use other langchain agents as tools. use our gpt index to find the necessary code samples"
print("input_query...")
res = run_agent_executor(agent, agent_tools, input_query, "i want a recursive langchain agent that can create an army of other agents to work for it")
# response = agent.execute(input_query, input_query)
print(res)
