import os
import shutil
import requests
import glob
from typing import Literal, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# --- Configuration & Paths ---
CHROMA_PATH = "./chroma_db_nigerian_legal_rag"
SOURCE_DOCS_DIR = "./source_docs"  # <--- Place your PDFs here
os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)

class LegalAgentApplication:
    def __init__(self):
        # 1. Initialize LLM
        self.api_key = os.getenv("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = ChatOpenAI(model="gpt-4o", api_key=self.api_key, temperature=0)
        
        # 2. Initialize Embeddings & Vector Store
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=self.api_key
        )
        self.vectorstore = Chroma(
            collection_name="nigerian_tenancy_docs",
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        # 3. Setup Tools
        self.browser = DuckDuckGoSearchResults(name="duckduckgo_search", 
                                             description="Search DuckDuckgo for recent Nigerian legal precedents or news")
        self.tools = self.create_tools()
        self.model_with_tools = self.model.bind_tools(self.tools)

        # 4. Setup Graph & Memory
        self.memory = MemorySaver()
        self.graph = self.build_graph()

        # 5. AUTO-LOAD: Ingest PDFs from source_docs on startup
        self.check_and_load_static_documents()

    def check_and_load_static_documents(self):
        """
        Automatically ingests any PDFs found in ./source_docs on startup.
        """
        pdf_files = glob.glob(os.path.join(SOURCE_DOCS_DIR, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDFs found in '{SOURCE_DOCS_DIR}'. Please add your Tenancy Law PDF there.")
            return

        print(f"Found {len(pdf_files)} hardcoded documents. Processing...")
        
        count = 0
        for pdf_path in pdf_files:
            try:
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                
                # Add to Vector Database
                if splits:
                    self.vectorstore.add_documents(splits)
                    count += len(splits)
                    print(f" Ingested: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f" Failed to load {pdf_path}: {e}")
        
        if count > 0:
            print(f"Auto-load complete! Added {count} chunks to the knowledge base.")
        else:
            print("No new chunks added (files might be empty or unreadable).")

    def create_tools(self):
        """
        Define tools tailored for a Nigerian Rental Law Assistant.
        """
        
        # --- Tool 1: General Utility ---
        @tool
        def get_weather(city: str) -> str:
            """
            Returns current weather. Useful if a tenant claims flooding or storm damage.
            Usage: get_weather("Lagos")
            """
            weather_api = os.getenv("WEATHER_API_KEY")
            print(weather_api)
            endpoint = f"https://api.weatherapi.com/v1/current.json?key={weather_api}&q={city}"
            
            try:
                res = requests.get(endpoint, timeout=5).json()
                return f"Weather in {res['location']['name']}: {res['current']['condition']['text']}, {res['current']['temp_c']}°C"
            except Exception as e:
                return f"Failed to fetch weather. Error: {e}"

        # --- Tool 2: Nigerian Legal Dictionary ---
        NIGERIAN_LEGAL_DICTIONARY = {
            "quit notice": "A statutory notice given by a landlord to a tenant to recover possession of premises (e.g., 7 days for weekly, 6 months for yearly tenancy).",
            "mesne profits": "Compensation claimed by a landlord against a tenant who wrongfully remains in the property after their tenancy has expired.",
            "tenancy at will": "A tenancy that can be terminated at any time by either the landlord or tenant, usually where no rent is fixed or paid.",
            "covenant of quiet enjoyment": "The landlord's implied promise not to interfere with the tenant's possession and use of the property.",
            "security deposit": "Caution fee paid by the tenant to cover potential damages, usually refundable at the end of the tenancy.",
            "rent advance": "Payment of rent for a future period. In Lagos, it is unlawful to demand more than 1 year from a new yearly tenant."
        }

        @tool
        def dictionary_lookup(term: str) -> str:
            """
            Look up definitions of Nigerian tenancy law terms.
            Usage: dictionary_lookup("quit notice")
            """
            key = term.lower().strip()
            return NIGERIAN_LEGAL_DICTIONARY.get(
                key,
                f"No Nigerian legal definition found for '{term}'. Try a different term."
            )

        # --- Tool 3: Document Retrieval (RAG Tool) ---
        @tool
        def retrieve_documents(query: str) -> str:
            """
            Search the uploaded legal database (e.g., Lagos Tenancy Law 2011).
            Use this when the user asks about specific contract details, notice periods, or sections of the law.
            """
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
            results = retriever.invoke(query)
            
            if not results:
                return "No relevant legal documents found."
            
            return "\n\n---\n\n".join(
                f"Document Clause/Section {i+1}:\n{doc.page_content}" for i, doc in enumerate(results)
            )

        return [get_weather, dictionary_lookup, retrieve_documents, self.browser]

    def build_graph(self):
        """
        Builds the decision-making graph.
        """
        
        # Node: Access Model
        def access_model(state: MessagesState) -> MessagesState:
            sys_message = """
            You are a specialized Nigerian Tenancy Law Assistant.
            
            YOUR GOAL: Provide accurate information based on the Lagos State Tenancy Law 2011 and Nigerian Property Law.
            
            DECISION RULES:
            1. **General Chat**: If the user says "Hello", answer directly.
            2. **Definitions**: If the user asks "What does 'Quit Notice' mean?", use `dictionary_lookup`.
            3. **Specific Cases/Docs**: If the user asks about specific lease terms, notice periods, or rights under the Lagos Tenancy Law, you MUST use `retrieve_documents`.
            4. **External Search**: Use the browser for recent Nigerian court judgments.

            GUIDELINES:
            - Always assume amounts are in **Naira (₦)**.
            - Refer to "Quit Notices" rather than "eviction notices".
            
            DISCLAIMER: Always clarify you are an AI assistant, not a Lawyer.
            """
            response = self.model_with_tools.invoke(
                [SystemMessage(content=sys_message)] + state["messages"]
            )
            return {"messages": [response]}

        # Edge Logic
        def should_continue(state: MessagesState):
            if state["messages"][-1].tool_calls:
                return "tools"
            else:
                return "exit"

        # Build Graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("access_model", access_model)
        workflow.add_node("tool_node", ToolNode(self.tools))

        workflow.add_edge(START, "access_model")
        workflow.add_conditional_edges("access_model", should_continue, {"tools": "tool_node", "exit": END})
        workflow.add_edge("tool_node", "access_model")

        return workflow.compile(checkpointer=self.memory)

    def query_agent(self, query: str, thread_id: str = "client_case_001"):
        config = {"configurable": {"thread_id": thread_id}}
        response = self.graph.invoke(
            {"messages": [HumanMessage(content=query)]}, 
            config=config
        )
        return response["messages"][-1].content

# --- FastAPI Implementation ---

app = FastAPI(title="Nigerian Rental Law AI Assistant")
legal_agent = LegalAgentApplication()

class CaseQuery(BaseModel):
    query: str = Field(..., example="What is the required notice period for terminating a yearly tenancy in Lagos?")
    case_id: str = "default_case"

@app.post("/consult")
async def consult_agent(request: CaseQuery):
    try:
        response = legal_agent.query_agent(request.query, request.case_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Standard way to run a standalone script
    uvicorn.run(app, host="127.0.0.1", port=8000)