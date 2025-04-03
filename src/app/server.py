from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from pydantic import BaseModel
import asyncio
import time
import threading
import uuid
from src.agents.agent_router import AgentRouter
from src.agents.product_advisor import ProductAdvisorAgent
from src.agents.policy_advisor import PolicyAdvisorAgent
from src.agents.general_advisor import GeneralAdvisorAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is starting up...")
    # Initialize global agents that will be reused
    app.state.agent_router = AgentRouter()

    # Đảm bảo khóa trong app.state.agents giống với AgentRouter.agent_types
    app.state.agents = {
        "product_advisor": ProductAdvisorAgent(),
        "policy_advisor": PolicyAdvisorAgent(),
        "general": GeneralAdvisorAgent(),
        # Thêm các agent khác nếu có
    }

    # Set the default agent
    app.state.default_agent = app.state.agents["general"]
    yield
    print("App is shutting down...")


app = FastAPI(title="TechPlus Hardware Advisor API", lifespan=lifespan)
chat_sessions = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str = None
    language: str = "vi"


class QueryResponse(BaseModel):
    response: str
    session_id: str


class ChatSession:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.messages = []
        self.last_activity = time.time()
        self.lock = threading.Lock()
        self.response_ready = False
        self.current_response = None


async def cleanup_old_sessions():
    while True:
        current_time = time.time()
        # Sessions timeout after 30 minutes of inactivity
        sessions_to_remove = [
            session_id for session_id, session in chat_sessions.items()
            if current_time - session.last_activity > 1800  # 30 minutes
        ]

        for session_id in sessions_to_remove:
            del chat_sessions[session_id]

        # Check every minute
        await asyncio.sleep(60)


@app.get("/")
def read_root():
    return {"message": "TechPlus Hardware Advisor API"}


async def process_chat_query(app: FastAPI, session_id: str, query: str, language: str = "vi"):
    try:
        session = chat_sessions[session_id]

        # Step 1: Route the query to the appropriate agent type
        agent_type = await app.state.agent_router.route_query(query)
        print("Agent type determined:", agent_type)
        print(f"Routing query to agent type: {agent_type}")
        print(f"Available agents: {list(app.state.agents.keys())}")

        # Step 2: Get the appropriate agent
        agent_instance = app.state.agents.get(agent_type)
        if not agent_instance:
            # Fall back to general advisor if the specific agent type isn't implemented
            print(
                f"Agent type {agent_type} not found in app.state.agents, falling back to general advisor")
            agent_instance = app.state.default_agent
        else:
            print(f"Using agent: {agent_instance.agent.name}")

        # Step 3: Handle the query with the selected agent
        response_content = await agent_instance.handle_query(query, language)

        # Step 4: Store the response
        with session.lock:
            # Add to session messages
            session.messages.append({
                "role": "user",
                "content": query
            })

            # Create response object with agent metadata
            agent_response = {
                "content": response_content,
                "sender": agent_instance.agent.name
            }

            # Add the response to messages
            session.messages.append({
                "role": "assistant",
                "content": response_content,
                "agent_responses": [agent_response]
            })

            # Mark response as ready for polling
            session.response_ready = True
            session.current_response = response_content
            session.last_activity = time.time()

            return response_content

    except Exception as e:
        error_message = f"Lỗi khi xử lý truy vấn: {str(e)}"
        print(error_message)
        if session_id in chat_sessions:
            session = chat_sessions[session_id]
            with session.lock:
                session.messages.append({
                    "role": "user",
                    "content": query
                })
                session.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                session.response_ready = True
                session.current_response = error_message
        return error_message


@app.post("/ask", response_model=QueryResponse)
async def ask_query(request: QueryRequest, background_tasks: BackgroundTasks):
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession()
        chat_sessions[session_id].id = session_id

    session = chat_sessions[session_id]
    session.last_activity = time.time()
    session.response_ready = False

    # Process query in background
    background_tasks.add_task(
        process_chat_query,
        app,
        session_id,
        request.query,
        request.language
    )

    # Return immediate acknowledgment
    return {
        "response": "Đang xử lý câu hỏi của bạn. Vui lòng đợi trong giây lát...",
        "session_id": session_id
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "messages": session.messages,
        "last_activity": session.last_activity
    }


@app.get("/poll/{session_id}")
async def poll_session(session_id: str):
    """
    Poll server for response. This function is called repeatedly by the client to check for updates.
    It needs to be fast and responsive to avoid timeouts and provide a good user experience.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = chat_sessions[session_id]
    with session.lock:
        # Check if we have a new response ready
        has_response = session.response_ready

        # Return the latest messages
        if len(session.messages) >= 2:
            last_user_msg = None
            last_assistant_msg = None

            # Find the most recent user and assistant messages
            for msg in reversed(session.messages):
                if msg["role"] == "user" and last_user_msg is None:
                    last_user_msg = msg
                elif msg["role"] == "assistant" and last_assistant_msg is None:
                    last_assistant_msg = msg

                if last_user_msg is not None and last_assistant_msg is not None:
                    break

            # Once client has received the response, reset the response_ready flag
            if has_response:
                session.response_ready = False
                session.current_response = None

            return {
                "session_id": session_id,
                "has_response": has_response,
                "user_message": last_user_msg,
                "assistant_message": last_assistant_msg
            }
        else:
            return {
                "session_id": session_id,
                "has_response": False
            }
