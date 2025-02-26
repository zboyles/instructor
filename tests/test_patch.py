from pydantic import BaseModel
import pytest
import openai
from instructor import patch


@pytest.mark.skip("Not implemented")
def test_runmodel():
    patch()

    class UserExtract(BaseModel):
        name: str
        age: int

    model = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"


@pytest.mark.skip("Not implemented")
def test_runmodel_validator():
    patch()

    from pydantic import field_validator

    class UserExtract(BaseModel):
        name: str
        age: int

        @field_validator("name")
        @classmethod
        def validate_name(cls, v):
            if v.upper() != v:
                raise ValueError("Name should be uppercase")
            return v

    model = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"
