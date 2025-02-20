from pydantic import BaseModel, Field

class PostContent(BaseModel):
    """""Validate response and provide feedback"""""
    ridiculous: bool = Field(description="Is the post utterly ridiculous/completely inappropriate? ")
    leaks_pii: bool = Field(description="Does the post expose any sensitive Personally Identifiable Information? ")
    relevant_to_category: bool = Field(description="Is the content even remotely related to post category? Be fairly lenient with this.")
