from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path
import os


class OpenAIConfig(BaseModel):
    api_key: str = Field(default=os.getenv("OPENAI_API_KEY"), description="open ai api key")
    model_name: str = Field(default="gpt-4o-mini", description = "chatgpt model to be used")
    prompt_template: str = Field ( """
# Instructions:

You will be provided with a message/email. Your task is to determine whether the message/email is spam or not spam.

# Output:

- If the message is spam, output: **1**
- If the message is not spam, output: **0**

**Important Guidelines:**

- Your output must be a single digit: either **"1"** or **"0"**.
- Do **not** include any additional text, explanations, or formatting.
- Do **not** repeat the message.
- Do **not** include any other text in your response.

# Examples:

Example 1:

Message:
"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question (std txt rate) T&C's apply 08452810075over18's"

Output:
1

---

Example 2:

Message:
"U dun say so early hor... U c already then say..."

Output:
0

---

Example 3:

Message:
"Nah I don't think he goes to usf, he lives around here though"

Output:
0

---

Example 4:

Message:
"FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun—you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"

Output:
1

""", description="Prompt template for spam vs ham determination"
    )


class Message(BaseModel):
    content: str = Field(..., description="message that is to be classified")

class Openaioutput(BaseModel):
    prediction: int = Field(..., description = "message output label")


def OpenAIModel(input_message:Message, config:OpenAIConfig) -> Openaioutput:
    client = OpenAI(api_key=config.api_key)
    
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[
            {
            "role": "system",
            "content": config.prompt_template
            },
            {
            "role": "user",
            "content": input_message
            }
        ],
        temperature=0,
        max_tokens=64,
        top_p=1
        )
    prediction = response.choices[0].message.content.strip()
    return Openaioutput(prediction=prediction)