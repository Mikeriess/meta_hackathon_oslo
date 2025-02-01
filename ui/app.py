import chainlit as cl
import ollama

@cl.on_chat_start
async def start():
    # Send initial message
    await cl.Message(
        content="Welcome! Please send an image and I'll analyze it for you."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Get image elements from the message
    image_elements = message.elements
    
    if not image_elements:
            response = ollama.chat(
                model='x/llama3.2-vision',  # Update model name as needed
                messages=[
                    {
                        'role': 'user',
                        'content': 'What is in this image?',
                    },
                ],
            )
            
            # Display the image and analysis
            await cl.Message(
                content=response['message']['content'],

            ).send()
        
    # Process each image element
    for image in image_elements:
        try:
            # Get image content
            if image.path:
                with open(image.path, 'rb') as file:
                    image_data = file.read()
            else:
                image_data = image.content
        
            # Send image to Ollama for analysis
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
            
            # Display the image and analysis
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