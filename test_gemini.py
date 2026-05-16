# test_gemini.py
import os
from google import genai

client = genai.Client(api_key="AIzaSyCth-ZlAn8dEEaCIyk82I2-9oi368EGQLg")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say exactly: setup works"
)

print(response.text)