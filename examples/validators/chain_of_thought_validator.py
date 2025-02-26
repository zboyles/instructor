import instructor
import openai
from pydantic import BaseModel, Field, model_validator
from typing import Optional

# Enables `response_model` and `max_retries` parameters
instructor.patch()


class Validation(BaseModel):
    is_valid: bool = Field(
        ..., description="Whether the value is valid given the rules"
    )
    error_message: Optional[str] = Field(
        ...,
        description="The error message if the value is not valid, to be used for re-asking the model",
    )


def validator(values):
    chain_of_thought = values["chain_of_thought"]
    answer = values["answer"]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value is valid for the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Verify that `{answer}` follows the chain of thought: {chain_of_thought}",
            },
        ],
        # this comes from instructor.patch()
        response_model=Validation,
    )
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return values


class Response(BaseModel):
    chain_of_thought: str
    answer: str

    @model_validator(mode="before")
    @classmethod
    def chain_of_thought_makes_sense(cls, data):
        return validator(data)


if __name__ == "__main__":
    try:
        resp = Response(
            chain_of_thought="1 + 1 = 2", answer="The meaning of life is 42"
        )
        print(resp)
    except Exception as e:
        print(e)
        """
        1 validation error for Response
            Value error, The statement 'The meaning of life is 42' does not follow the chain of thought: 1 + 1 = 2. 
            [type=value_error, input_value={'chain_of_thought': '1 +... meaning of life is 42'}, input_type=dict]
        """
