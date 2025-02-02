import chainlit as cl
import httpx
import base64

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"  # Update if needed
MODEL_NAME = "your-model-name"  # Replace with the actual model name


@cl.on_chat_start
async def start():
    # Initialize session history
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome! Test out Llama3.2-Vision-MykMaks!").send()


async def query_vllm(messages):
    """Send a request to the vLLM API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            VLLM_API_URL,
            json={"model": MODEL_NAME, "messages": messages},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


@cl.on_message
async def main(message: cl.Message):
    try:
        history = cl.user_session.get("history")
        image_elements = message.elements

        if image_elements:
            for image in image_elements:
                try:
                    if image.path:
                        with open(image.path, "rb") as file:
                            image_data = file.read()
                    else:
                        image_data = image.content

                    # Convert image to base64 if vLLM supports images
                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                    messages = history + [
                        {
                            "role": "user",
                            "content": "What is in this image?",
                            "images": [image_base64],  # Adjust based on vLLM API
                        }
                    ]

                    response = await query_vllm(messages)
                    assistant_message = response["choices"][0]["message"]["content"]

                    history.append({"role": "user", "content": "What is in this image?", "images": [image_base64]})
                    history.append({"role": "assistant", "content": assistant_message})
                    cl.user_session.set("history", history)

                    await cl.Message(
                        content=assistant_message,
                        elements=[
                            cl.Image(
                                name="Analyzed Image",
                                content=image_data,
                                display="inline"
                            )
                        ]
                    ).send()
                except Exception as e:
                    await cl.Message(content=f"Error processing image: {str(e)}").send()
        else:
            messages = history + [{"role": "user", "content": message.content}]
            response = await query_vllm(messages)
            assistant_message = response["choices"][0]["message"]["content"]

            history.append({"role": "user", "content": message.content})
            history.append({"role": "assistant", "content": assistant_message})
            cl.user_session.set("history", history)

            await cl.Message(content=assistant_message).send()

    except Exception as e:
        await cl.Message(content=f"Error processing message: {str(e)}").send()
