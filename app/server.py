from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
import asyncio
import time
import threading
import uuid
from ..agents.agent_router import AgentRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is starting up...")
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


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_sessions())


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


def process_chat_query(session_id: str, query: str):
    try:
        session = chat_sessions[session_id]
        with session.lock:
            # Initialize a response collector
            response_collector = []

            # Use a callback to collect the agent's response
            def response_callback(content: str, sender):
                if sender != "CustomerService":  # Don't include user proxy messages
                    response_collector.append({
                        "content": content,
                        "sender": sender
                    })

            # Extract the general agent from the router
            agents = AgentRouter()
            general_agent = agents.agent_types["general"]

            # Set the response callback
            for agent in agents.agent_types:
                if hasattr(agent, "register_reply_callback"):
                    agent.register_reply_callback(response_callback)

            # Initiate the chat with the general agent
            general_agent.initiate_chat(
                agents["manager"],
                message=query,
                clear_history=False
            )

            # Process all responses and combine them
            if response_collector:
                combined_response = ""
                for response_item in response_collector:
                    agent_name = response_item["sender"]
                    content = response_item["content"]

                    if combined_response:
                        combined_response += f"\n\n"

                    combined_response += content

                # Add to session messages
                session.messages.append({
                    "role": "user",
                    "content": query
                })
                session.messages.append({
                    "role": "assistant",
                    "content": combined_response,
                    "agent_responses": response_collector
                })

                return combined_response
            else:
                error_message = "Không nhận được phản hồi từ các chuyên gia. Vui lòng thử lại."
                session.messages.append({
                    "role": "user",
                    "content": query
                })
                session.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                return error_message

    except Exception as e:
        error_message = f"Lỗi khi xử lý truy vấn: {str(e)}"
        print(error_message)
        if session_id in chat_sessions:
            session = chat_sessions[session_id]
            session.messages.append({
                "role": "user",
                "content": query
            })
            session.messages.append({
                "role": "assistant",
                "content": error_message
            })
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

    # Process query in background
    background_tasks.add_task(process_chat_query, session_id, request.query)

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
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = chat_sessions[session_id]
    with session.lock:
        # Return the latest messages
        if len(session.messages) >= 2:
            last_user_msg = None
            last_assistant_msg = None

            # Find the last user and assistant messages
            for msg in reversed(session.messages):
                if msg["role"] == "user" and last_user_msg is None:
                    last_user_msg = msg
                elif msg["role"] == "assistant" and last_assistant_msg is None:
                    last_assistant_msg = msg

                if last_user_msg is not None and last_assistant_msg is not None:
                    break

            return {
                "session_id": session_id,
                "has_response": True,
                "user_message": last_user_msg,
                "assistant_message": last_assistant_msg
            }
        else:
            return {
                "session_id": session_id,
                "has_response": False
            }
