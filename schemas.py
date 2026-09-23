from pydantic import BaseModel, field_validator
from typing import List
from enum import Enum 

class CriticalityLevel(str, Enum):
    BAJA="baja"
    MEDIA="media"
    ALTA="alta"

class TechnicalExtraction(BaseModel):
    tecnologias: List[str]
    nivel_de_criticidad: CriticalityLevel
    resumen_tecnico: str
    @field_validator("tecnologias")
    @classmethod
    def technologies_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("technologies must contain at least one item")
        return value