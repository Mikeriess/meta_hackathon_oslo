import chainlit as cl
import ollama

@cl.on_chat_start
async def start():
    # Send initial message
    await cl.Message(
        content="Welcome! How can I assist you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        image_elements = message.elements
        
        if image_elements:
            for image in image_elements:
                try:
                    if image.path:
                        with open(image.path, 'rb') as file:
                            image_data = file.read()
                    else:
                        image_data = image.content
                    
                    response = ollama.chat(
                        model='x/llama3.2-vision',  # Update model name as needed
                        messages=[
                            {
                                'role': 'user',
                                'content': 'What is in this image?',
                                'images': [image_data],
                            },
                        ],
                    )
                    
                    await cl.Message(
                        content=response['message']['content'],
                        elements=[
                            cl.Image(
                                name="Analyzed Image",
                                content=image_data,
                                display="inline"
                            )
                        ]
                    ).send()
                except Exception as e:
                    await cl.Message(
                        content=f"Error processing image: {str(e)}"
                    ).send()
        else:
            response = ollama.chat(
                model='x/llama3.2-vision',  # Update model name as needed
                messages=[
                    {
                        'role': 'user',
                        'content': message.content,
                    },
                ],
            )
            
            await cl.Message(
                content=response['message']['content']
            ).send()
    
    except Exception as e:
        await cl.Message(
            content=f"Error processing message: {str(e)}"
        ).send()
