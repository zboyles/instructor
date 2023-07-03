from fastapi import FastAPI
from openai_function_call import OpenAISchema
from openai_function_call.dsl import MultiTask
from openai_function_call.dsl import ChatCompletion, messages as m
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Example Application using openai_function_call",
    description="This API accepts natural language queries and returns lists of queries and citations",
    version="0.0.1",
)


class Citation(BaseModel):
    substring_quote: List[str] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content",
    )


class SearchRequest(BaseModel):
    body: str


class SearchQuery(OpenAISchema):
    title: str = Field(..., description="Question that the query answers")
    query: str = Field(
        ...,
        description="Detailed, comprehensive, and specific query to be used for semantic search",
    )


SearchResponse = MultiTask(
    subtask_class=SearchQuery,
    description="Correctly segmented set of search queries",
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    task = (
        ChatCompletion(name="Segmenting Search requests")
        | m.ExpertSystem("Segmented search queries into multiple queries")
        | m.TaggedMessage(content=request.body, tag="query")
        | m.TipsMessage(
            tips=[
                "Expand query to contain multiple forms of the same word (SSO -> Single Sign On)",
                "Use the title to explain what the query should return, but use the query to complete the search",
                "The query should be detailed, specific, and cast a wide net when possible",
            ]
        )
        | SearchRequest
    )
    return await task.acreate()
