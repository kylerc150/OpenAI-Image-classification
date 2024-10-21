#Imports needed for project
import numpy
import cv2
import base64
import requests
import gradio
from PIL import Image

#API key needed from your account
API_KEY = ""

# Encodes out image so Gpt-4o can read it
def encodeImage(image):
    
    # Flips the image horizontally
    imageFlip = numpy.fliplr(image)
    
    # Converts the flipped image form RGB to BGR color - OpenCV uses BGR
    imageColor = cv2.cvtColor(imageFlip, cv2.COLOR_RGB2BGR)
    
    # encoded to jpg format
    success, buffer = cv2.imencode('.jpg', imageColor)
    
    # converts the image to a Base 64 string
    encodedImage = base64.b64encode(buffer).decode("utf-8")
    
    # return the finalized image
    return encodedImage

# asks our prompt for our model
def buildPrompt(classes):
    
    # question we are asking
    prompt = f"What is in the image? Return the class of object in the image for classes {','.join(classes)}. You can only return one class."
    
    # return the question
    return prompt

# request sent to OpenAI's API
def buildRequest(prompt, image):
    
    # prepares authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # data sent by the sender in packets
    payload = {
        # model we are using
        "model": "gpt-4o",
        
        # What the model is going to look at
        "messages": [
            {
                # we are the user roles
                "role": "user",
                
                # This contains image to look at
                "content": [
                    {
                        # we want to show the API text
                        "type": "text",
                        #show it our prompt
                        "text": prompt
                    },
                    {
                        # we are give the API an Image
                        "type": "image_url",
                        # our image
                        "image_url": {
                            #image in base64
                            "url": f"data:image/jpeg;base64,{image}" 
                        }
                    }
                ]
            }
        ]
    }
    
    # sends a POST method to OpenAI API's chat/completions endpoint and returns a response
    response = requests.post(url= "https://api.openai.com/v1/chat/completions",
                  headers = headers,
                  json = payload)
    
    response = response.json()
    
    return response


# Take image input, encodes it, and builds a prompt with specific classes (rabbit, dog, bird)
# Sends image and prompt to the openAI API using buildRequest and returns the model's classification result.
def classifyImage(image):
    
    encodedImage = encodeImage(image)
    
    prompt = buildPrompt(classes = ["rabbit", "dog", "bird"])
    
    response = buildRequest(prompt, encodedImage)
    
    return response["choices"][0]["message"]["content"]

# Makes a gradio interface    
app = gradio.Interface(
    fn = classifyImage,
    inputs = gradio.Image(type="numpy"),
    outputs = "text",
    title = "Image Classification with GPT 4 Vision"
)

#launches the gradio interface
app.launch()
