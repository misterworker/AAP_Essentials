from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
import asyncio, os, random

load_dotenv()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://work-advisor.vercel.app/post-prediction"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

OpenAI_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

nemo_nvidia_llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    temperature=0.5,
    max_tokens=500,
    api_key=NVIDIA_API_KEY,
)

class PostContent(BaseModel):
    """""Validate response and provide feedback"""""
    ridiculous: bool = Field(description="Is the post utterly ridiculous/completely inappropriate? ")
    leaks_pii: bool = Field(description="Does the post expose any sensitive Personally Identifiable Information? ")
    relevant_to_category: bool = Field(description="Is the content even remotely related to post category? Be very lenient with this.")

OpenAI_bot = OpenAI_llm.with_structured_output(PostContent, method="function_calling")
NVIDIA_bot = nemo_nvidia_llm.with_structured_output(PostContent, method="function_calling")
    
@app.post("/validate_post")
async def validate_post(request: Request):
    data = await request.json()
    content = data.get("content", "")
    category = data.get("category", "")

    if not content:
        return JSONResponse(
            content={"error": "Content not found"},
            status_code=400
        )
    
    if not category:
        return JSONResponse(
            content={"error": "Category not found"},
            status_code=400
        )

    try:
        data = f"Post Category: {category}\nPost content:\n{content}"
        prompt = (f"You are a bot that detects suitability of the post content for public posting.\n{data}")
        random_choice = random.randint(1, 2)
        match random_choice:
            case (1): bot = OpenAI_bot
            case (2): bot = NVIDIA_bot
        result = await asyncio.to_thread(NVIDIA_bot.invoke, prompt)
        print("Result: ", result)

        try:
            # Create tasks for each coroutine
            prompt = ("Based on the post category, content and analysis results provided, create recommendations on where the user can improve on the results, "
            f"limited to a single paragraph.\nCategory: {category}\ncontent: {content}\nAnlysis result: {result}")
            messages = [HumanMessage(content=prompt)]

            async def get_nemo_response():
                await asyncio.sleep(10)
                print("NVIDIA")
                try:
                    response = await nemo_nvidia_llm.ainvoke(messages)
                    return response
                except:
                    await asyncio.sleep(11)

            async def get_openai_response():
                try:
                    response = await OpenAI_llm.ainvoke(messages)
                    print("Response1", response)
                    return response
                except:
                    await asyncio.sleep(21)

            nemo_task = asyncio.create_task(get_nemo_response())
            openai_task = asyncio.create_task(get_openai_response())
            done, pending = await asyncio.wait([openai_task, nemo_task], return_when=asyncio.FIRST_COMPLETED, timeout=20)

            for task in done: response = task.result(); break
            for task in pending: task.cancel()

        except asyncio.TimeoutError:
            # If neither task completes within 20 seconds, return fallback response
            print("Both tasks took too long. Returning fallback response.")
            response.content = "Response took too long. Sorry about that. Please try again."

        print("Content: ", response.content)
            
        if not response.content: response.content = "Response unavailable. Sorry😛. Error te-0"
        return JSONResponse(content={"response": response.content, "result": dict(result)})
    
    except Exception as e:
        print("error: ", e)
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )
    
