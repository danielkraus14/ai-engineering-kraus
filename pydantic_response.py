import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# TODO 1: Define la clase Pydantic 'EntityExtraction' 
# Debe tener: topic (str), entities (Lista de str), y sentiment_score (float entre 0 y 1)
class EntityExtraction(BaseModel):
    topic: str
    entities: List[str]
    sentiment_score: float = Field(ge=0, le=1)

async def run_validated_chain(text: str):
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    
    # TODO 2: Configura el modelo para usar la salida estructurada con Pydantic
    # Tip: Usa el método .with_structured_output()
    structured_llm = llm.with_structured_output(EntityExtraction)
    
    # TODO 3: Agrega una estrategia de reintento con .with_retry() 
    # para que sea resiliente ante fallos de conexión (máximo 3 intentos).
    resilient_llm = structured_llm.with_retry(stop_after_attempt=3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el texto y extrae las entidades."),
        ("human", "{input}")
    ])
    
    # TODO 4: Une el prompt con el resilient_llm y ejecuta asíncronamente
    # No olvides manejar excepciones con try/except para capturar fallos de validación
    resilient_chain = prompt | resilient_llm
    try:
        result = await resilient_chain.ainvoke({
            "input": text
        })
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    sample_text = "LangGraph es una extensión de LangChain para agentes cíclicos."
    asyncio.run(run_validated_chain(sample_text))
