import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
gemini_api_key = os.environ.get("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash",google_api_key=gemini_api_key, temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert culinary assistant. Given an ingredient or dish name, suggest a clear, easy-to-follow recipe with a short list of ingredients and numbered preparation steps. Keep your answer concise and practical."),
    ("human", "I have the following ingredient or dish in mind: {request}")
])

parser = StrOutputParser()

chain = prompt | model | parser

async def main():
    result = await chain.ainvoke(
        {"request": "pasta carbonara"}
    )
    print(result)
    
if __name__ == "__main__":
    asyncio.run(main())