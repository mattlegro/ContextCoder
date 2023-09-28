import streamlit as st
import tempfile
import os

from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions import create_conversational_retrieval_agent
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler

@st.cache_resource
def generate_retriever(files,openai_api_key):        

    with tempfile.TemporaryDirectory() as tmp_dir_name:

        for file in files:
            with tempfile.NamedTemporaryFile(delete=False,dir=tmp_dir_name) as tmp_file:
                tmp_file.write(file.read())
                
        loader = GenericLoader.from_filesystem(
                tmp_dir_name,
                glob="**/*",
                suffixes=None,
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500))
            
        docs = loader.load()

        python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                   chunk_size=2000, 
                                                                   chunk_overlap=200)
        texts = python_splitter.split_documents(docs)
        
        db = Chroma.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai_api_key,disallowed_special=()))
        retriever = db.as_retriever(search_type="mmr", search_kwargs = {"k": 8})
        
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# @st.cache_resource
def create_agent_chain(_llm,_retriever,serpapi_api_key):
                
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        tools = [
            create_retriever_tool(
                name="Uploaded Files",
                retriever=_retriever,
                description="The user has uploaded files. Anytime you need to reference those files reference this. Input should be a fully formed question.",
            ),
            Tool(
                name="Google Search",
                func=search.run,
                description="Use to search the internet for relevant information when you do not know the answer, or to provide additional specifics or context.",
            ),
        ]

        system_message = None
        agent = create_conversational_retrieval_agent(tools, _llm, verbose=True, system_message=system_message,remember_intermediate_steps=True)
        
        return agent

# @st.cache_resource
def load_llm(openai_api_key):
    llm = ChatOpenAI(model_name="gpt-4",temperature=0.2,openai_api_key=openai_api_key)
    return llm

st.title("🦜🔗 Context Coderbot")

if st.secrets['OPENAI_API_KEY']:
    openai_api_key = st.secrets['OPENAI_API_KEY']
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

if st.secrets['SERPAPI_API_KEY']:
    serpapi_api_key = st.secrets['SERPAPI_API_KEY']
else:
    serpapi_api_key = st.sidebar.text_input("SerpAPI Key", type="password")

if not openai_api_key or not serpapi_api_key:
    st.info("Please add your OpenAI API key and your SerpAPI key to continue.")
    st.stop()

llm = load_llm(openai_api_key)

uploaded_files = st.file_uploader(label="Upload your .py files here!", accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload files to continue.")
    st.stop()

if uploaded_files:
    retriever = generate_retriever(uploaded_files,openai_api_key)
    st.write('Files Processed Successfully!')

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

agent_chain = create_agent_chain(llm,retriever,serpapi_api_key)
        
if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message("Hi! I'm ready to help you code. Ask a question so we can begin!")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if new_query := st.chat_input(placeholder='Write your messages here.'):
    st.chat_message('user').write(new_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = agent_chain.run(new_query, callbacks=[retrieval_handler, stream_handler])
        st.write(response)