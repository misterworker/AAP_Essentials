from pydantic import BaseModel, Field

class Suggestion(BaseModel):
    """""Suggest a new post title and post content"""""
    new_post_title: str = Field(description="Suggest a new post title")
    new_post_content: str = Field(description="Craft the new post content")