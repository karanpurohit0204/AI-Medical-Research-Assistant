import os
from dotenv import load_dotenv

file_loaded = load_dotenv()  
print(f"Was .env file loaded successfully? {file_loaded}")

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")  # Default voice ID
ncbi_api_key = os.getenv("NCBI_API_KEY")
ncbi_email = os.getenv("NCBI_EMAIL")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

print("Mistral API Key:", os.getenv("MISTRAL_API_KEY"))