import json
from fastapi import APIRouter, HTTPException, Request
import httpx
from pydantic import BaseModel
from typing import AsyncGenerator, List, Dict, Literal, Optional, Set, Any
from datetime import datetime
import asyncio
from enum import Enum
import openai  # or anthropic, depending on your LLM choice
from ollama import chat
from ollama import ChatResponse
from fastapi.responses import StreamingResponse

from speaches import kokoro_utils

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from textwrap import dedent
from langchain_core.messages import BaseMessage


class ConversationContext(BaseModel):
    current_topic: str
    user_level: int  # CEFR level: 1=A1, 2=A2, 3=B1, 4=B2, 5=C1, 6=C2
    recent_vocabulary: Set[str]
    grammar_patterns: Set[str]
    user_interests: List[str]
    target_language: str = "en"

    class Config:
        arbitrary_types_allowed = True


def _generate_system_prompt(context: ConversationContext) -> str:
    cefr_level = ["A1", "A2", "B1", "B2", "C1", "C2"][context.user_level - 1]

    return dedent(
        f"""You are a helpful language learning partner having a natural conversation. 

            Situation:
                - You are a seller in a coffe shop and the customer is asking about the menu.
                - Engage with the customer and provide information about the menu.
                - Ask him for different options of coffee and pastries.
                - Ask him if he wants to sit in or take away.
                - Ask him if he wants to pay by cash or card.
                - Ask him if he wants to add a tip.
                - Ask him if he wants a receipt.
                - Greet the user and thank him for his visit.

            Guidelines:
                - Use natural, conversational language appropriate for {cefr_level}
                - Keep responses concise and engaging, no more than a tweet length
                - Stay in context of the conversation
                - You can only answer back in {kokoro_utils.LanguageToText[context.target_language]}
                - Adapt your language to CEFR level {cefr_level}.
            """
    )


router = APIRouter(tags=["assistant"], prefix="/v1/assistant")


class UserInput(BaseModel):
    user_id: str
    messages: list


class NewConversation(BaseModel):
    user_id: str
    user_level: int
    interests: List[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ToolResult(BaseModel):
    tool_call_id: str
    output: str


class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "en"
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


tools = {
    "get_weather": lambda location: f"Weather in {location}: Sunny, 22°C",
    # Add more tools as needed
}


async def execute_tool_call(tool_call: ToolCall) -> str:
    """
    Execute a tool call and return the result
    """
    if tool_call.name in tools:
        try:
            return tools[tool_call.name](**tool_call.arguments)
        except Exception as e:
            return f"Error executing tool {tool_call.name}: {str(e)}"
    return f"Tool {tool_call.name} not found"


# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/chat"


async def generate_stream(
    request_data: ChatRequest,
) -> AsyncGenerator[str | list[str | dict], None]:
    """
    Generator function that streams responses from Ollama with proper error handling
    """
    system_prompt = _generate_system_prompt(
        ConversationContext(
            current_topic="general",
            user_level=1,
            recent_vocabulary=set(),
            grammar_patterns=set(),
            user_interests=["music", "movies"],
            target_language=request_data.language,
        )
    )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in request_data.messages:
        if msg.role.lower() == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    print(messages)

    llm = ChatOllama(
        model="phi4",  # or any model available in your Ollama instance
        temperature=request_data.temperature,
        num_predict=request_data.max_tokens,
        base_url="http://localhost:11434",  # adjust if your Ollama endpoint is different
    )

    # If tool definitions are provided, bind them to the model
    # if request_data.tools:
    #     llm = llm.bind_tools(request_data.tools)

    # Use the async streaming method from ChatOllama
    try:
        async for chunk in llm.astream(messages):
            # Each chunk is a BaseMessageChunk; yield its content.
            yield chunk.content
            await asyncio.sleep(0.01)  # mimic a small pause between chunks if desired
    except Exception as e:
        yield f"Error: {str(e)}"


# async def generate_stream(request_data: ChatRequest) -> AsyncGenerator[str, None]:
#     """
#     Generator function that streams responses from Ollama with proper error handling
#     """
#     async with httpx.AsyncClient() as client:
#         try:
#             ollama_messages = []

#             system_prompt = _generate_system_prompt(
#                 ConversationContext(
#                     current_topic="general",
#                     user_level=6,
#                     recent_vocabulary=set(),
#                     grammar_patterns=set(),
#                     user_interests=["music", "movies"],
#                     target_language=request_data.language,
#                 )
#             )
#             ollama_messages.append({"role": "system", "content": system_prompt})

#             for msg in request_data.messages:
#                 message_dict = {"role": msg.role, "content": msg.content}
#                 if msg.tool_calls:
#                     message_dict["tool_calls"] = [
#                         tool_call.dict() for tool_call in msg.tool_calls
#                     ]
#                 if msg.tool_call_id:
#                     message_dict["tool_call_id"] = msg.tool_call_id
#                 ollama_messages.append(message_dict)

#             ollama_request = {
#                 "model": "llama3.2",
#                 "messages": ollama_messages,
#                 "stream": True,
#                 "temperature": request_data.temperature,
#             }

#             if request_data.tools:
#                 ollama_request["tools"] = request_data.tools

#             if request_data.max_tokens:
#                 ollama_request["max_tokens"] = request_data.max_tokens

#             print(ollama_request)
#             async with client.stream(
#                 "POST", OLLAMA_API_URL, json=ollama_request, timeout=30.0
#             ) as response:
#                 if response.status_code != 200:
#                     error_msg = await response.aread()
#                     yield f"Error: HTTP {response.status_code} - {error_msg}\n"
#                     return

#                 async for line in response.aiter_lines():
#                     if line:
#                         try:
#                             data = json.loads(line)
#                             if "error" in data:
#                                 yield f"Error: {data['error']}\n"
#                                 return
#                             elif "message" in data:
#                                 yield data["message"]["content"]
#                             else:
#                                 yield " "
#                             await asyncio.sleep(0.01)
#                         except json.JSONDecodeError:
#                             yield f"Error parsing response: {line}\n"
#                             return

#         except httpx.TimeoutException:
#             yield "Error: Request to Ollama timed out\n"
#         except httpx.RequestError as e:
#             yield f"Error: Failed to connect to Ollama - {str(e)}\n"
#         except Exception as e:
#             yield f"Error: Unexpected error - {str(e)}\n"


@router.post("/chat")
async def chat(request_data: ChatRequest, request: Request):
    """
    Endpoint that streams chat responses from Ollama with proper headers
    """
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
    }

    async def event_stream():
        try:
            async for chunk in generate_stream(request_data):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(), headers=headers, media_type="text/event-stream"
    )


@router.get("/{user_id}/summary")
async def get_conversation_summary(user_id: str):
    if user_id not in conversation_manager.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    context = conversation_manager.conversations[user_id]

    return {
        "user_level": context.user_level,
        "current_topic": context.current_topic,
        "vocabulary_learned": list(context.recent_vocabulary),
        "message_count": len(context.message_history),
        "interests": context.user_interests,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
