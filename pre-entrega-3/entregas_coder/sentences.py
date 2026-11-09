"""Evaluation dataset: one technical concept, paraphrases and keyword traps."""

from typing import Dict, List

CONCEPT = "Despliegue de microservicios"

# Same concept, deliberately different vocabulary: none of these sentences
# contains the words "despliegue", "microservicios" or "servicio".
RELEVANT: Dict[str, str] = {
    "R1": "Cada componente pequeño de la aplicación se empaqueta en un contenedor "
          "Docker y se orquesta con Kubernetes para salir a producción.",
    "R2": "Publicamos los módulos independientes del sistema en el clúster mediante "
          "un pipeline de CI/CD que construye imágenes y las lanza sin tiempo de inactividad.",
    "R3": "El equipo libera en producción cada API autónoma por separado, usando "
          "contenedores y rollouts graduales.",
    "R4": "Llevar a producción una arquitectura distribuida en piezas desacopladas "
          "exige automatizar la entrega, el escalado y la reversión de versiones.",
    "R5": "Subimos al clúster en la nube cada pieza liviana y escalable por separado, "
          "con entregas automatizadas y sin detener el sistema.",
}

# Traps: they share surface keywords with the concept but are irrelevant
# (T1, homonyms) or state the opposite position (T2).
TRAPS: Dict[str, str] = {
    "T1": "El servicio de micro-limpieza es excelente: despliegan un equipo de "
          "personal en cada oficina todas las mañanas.",
    "T2": "Nuestro monolito se despliega como un único bloque; los microservicios "
          "no aportan nada y rechazamos su despliegue.",
}


def corpus() -> Dict[str, str]:
    """Return all candidate sentences keyed by id (relevant first, then traps)."""
    return {**RELEVANT, **TRAPS}


def ids() -> List[str]:
    return list(corpus().keys())
