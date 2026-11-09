import re
import unicodedata
from typing import List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 200
OVERLAP_RATIO = 0.15  # overlap as a share of chunk_size, so the proportion survives resizing
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Control characters except tab/newline, plus invisible characters common in PDF text
# (zero-width space/joiners, BOM, soft hyphen).
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\u2060\ufeff\xad]")


class DocumentProcessor:
    def __init__(self,
                 model_encoding: str = "cl100k_base",
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: Optional[int] = None) -> None:
        self.tokenizer = tiktoken.get_encoding(model_encoding)
        self.chunk_size = chunk_size
        # Default overlap is a percentage of chunk_size unless given explicitly.
        self.chunk_overlap = int(chunk_size * OVERLAP_RATIO) if chunk_overlap is None else chunk_overlap

        # length_function makes chunk_size and chunk_overlap count tokens, not characters.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.calculate_tokens,
            separators=SEPARATORS,
        )

    def clean_text(self, text: str) -> str:
        """Remove PDF/web noise while keeping paragraph breaks.

        Paragraph breaks ("\\n\\n") are the splitter's preferred cut points, so flattening
        all whitespace would degrade chunk quality. NFC normalization comes first so that
        composed and decomposed accents become one representation (one token sequence,
        one embedding).
        """
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = CONTROL_CHARS.sub("", text)
        text = re.sub(r"[ \t]*(?:\.\s){2,}", ". ", text)     # ". . . ." leaders (tables of contents)
        text = re.sub(r"\.{4,}", "...", text)         # "........" runs, keep a plain ellipsis
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newline inside a paragraph
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Count tokens with the tiktoken encoder."""
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def process_document(self, raw_text: str) -> List[str]:
        """Pipeline: clean -> split. Empty chunks are dropped."""
        cleaned = self.clean_text(raw_text)
        return [c for c in self.splitter.split_text(cleaned) if c.strip()]


# Self-contained messy test text: duplicate spaces, stray newlines and blank-line runs.
# One paragraph is longer than chunk_size so the overlap between chunks is visible.
SAMPLE_TEXT = """La  generación aumentada por recuperación (RAG) combina un modelo de lenguaje con una base de conocimiento externa.   En lugar de confiar solo en lo que el modelo aprendió durante el entrenamiento, el sistema busca los fragmentos más relevantes de los documentos de la empresa y los incluye en el prompt. Así las respuestas se apoyan en información actualizada y verificable.



El primer paso de cualquier pipeline RAG es preparar los documentos.  Los textos que llegan desde PDFs, correos o páginas web suelen traer espacios duplicados,
saltos de línea en medio de las oraciones y bloques en blanco que no aportan significado.    Limpiar ese ruido antes de fragmentar mejora la calidad de los chunks y reduce el costo, porque cada token innecesario se paga al generar embeddings.

Un documento completo casi nunca cabe en la ventana de contexto de un modelo, y aunque cupiera, enviarlo entero sería caro y poco preciso. Por eso se divide en fragmentos. Un buen fragmento es lo bastante pequeño para ser específico y lo bastante grande para conservar el sentido de la idea que contiene. Si el corte cae en medio de una explicación, el significado se pierde y la búsqueda semántica devuelve resultados pobres.

La longitud de los fragmentos debe medirse en tokens y no en caracteres. Los modelos cobran, limitan y procesan texto en tokens, y la relación entre caracteres y tokens cambia según el idioma: el español, el código fuente o los números generan más tokens por carácter que el inglés. Medir en caracteres puede producir fragmentos que superan el límite del modelo de embeddings sin que nadie lo note.
El solapamiento, o overlap, repite el final de un fragmento al comienzo del siguiente. Su función es evitar que una idea quede partida justo en el límite entre dos chunks. Un overlap demasiado grande duplica información y encarece el índice; uno demasiado chico no protege el contexto. Un valor habitual ronda entre el diez y el quince por ciento del tamaño del fragmento.
El separador recursivo intenta cortar primero por párrafos, luego por líneas, después por oraciones y, como último recurso, por palabras. De esta forma respeta la estructura natural del texto siempre que sea posible. Cuando un párrafo es más largo que el tamaño máximo, el algoritmo baja de nivel y lo divide por oraciones, y es allí donde el solapamiento entre fragmentos se vuelve visible.

Una vez generados los fragmentos, cada uno se convierte en un vector mediante un modelo de embeddings y se guarda en una base vectorial. Cuando llega una consulta, se calcula su vector y se recuperan los fragmentos con mayor similitud coseno. La calidad de esos resultados depende directamente de cómo se limpió y se dividió el texto al principio del proceso."""


def shared_overlap(previous: str, following: str) -> str:
    """Longest suffix of `previous` that is also a prefix of `following`."""
    for size in range(min(len(previous), len(following)), 0, -1):
        if previous.endswith(following[:size]):
            return following[:size]
    return ""


def test_chunk_token_bounds(processor: DocumentProcessor, chunks: List[str]) -> None:
    """Every chunk must have 0 < tokens <= chunk_size."""
    assert chunks, "no chunks were produced"
    for i, chunk in enumerate(chunks):
        tokens = processor.calculate_tokens(chunk)
        assert 0 < tokens <= processor.chunk_size, (
            f"chunk {i} has {tokens} tokens, expected 1..{processor.chunk_size}")


def test_clean_text(processor: DocumentProcessor) -> None:
    """Deterministic checks for each kind of noise clean_text handles."""
    cases = {
        "duplicate spaces": ("a  b   c", "a b c"),
        "single newline joined": ("one\ntwo", "one two"),
        "paragraph break kept": ("one\n\n\n\ntwo", "one\n\ntwo"),
        "dot leaders (spaced)": ("Chapter 1 . . . . . 5", "Chapter 1. 5"),
        "dot leaders (run)": ("Chapter 1........ 5", "Chapter 1... 5"),
        "control chars": ("ab\x00c\x0cd\u200be", "abcde"),
        "unicode NFC": ("cafe\u0301", "caf\u00e9"),
    }
    for name, (raw, expected) in cases.items():
        result = processor.clean_text(raw)
        assert result == expected, f"{name}: got {result!r}, expected {expected!r}"
        print(f"clean_text OK | {name}")


def run_demo(text: str) -> None:
    processor = DocumentProcessor()
    cleaned = processor.clean_text(text)
    print(f"chunk_size={processor.chunk_size} chunk_overlap={processor.chunk_overlap}")
    print(f"Characters before/after cleaning: {len(text)} / {len(cleaned)}")
    print(f"Tokens before/after cleaning:     {processor.calculate_tokens(text)} / "
          f"{processor.calculate_tokens(cleaned)}")

    chunks = processor.process_document(text)
    print(f"Chunks generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"chunk={i} tokens={processor.calculate_tokens(chunk)} chars={len(chunk)} | {chunk[:60]!r}")
    test_chunk_token_bounds(processor, chunks)

    # Overlap evidence: text repeated at the boundary between consecutive chunks
    for i in range(len(chunks) - 1):
        shared = shared_overlap(chunks[i], chunks[i + 1])
        print(f"overlap {i}->{i + 1}: tokens={processor.calculate_tokens(shared)} | {shared[:70]!r}")


def build_long_text(text: str, copies: int = 3) -> str:
    """Repeat the text so larger chunk sizes split it, tagging each paragraph uniquely.

    Unique tags matter: with identical repeated paragraphs, shared_overlap would find
    false overlaps between copies and overstate the real overlap.
    """
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    tagged = [f"[Part {n}] {p}" for n, p in enumerate(paragraphs * copies, start=1)]
    return "\n\n".join(tagged)


def run_size_sweep(text: str, sizes=(200, 500, 800)) -> None:
    """Show the precision/context trade-off: bigger chunks mean fewer, broader pieces."""
    print("\nchunk_size sweep (same text, overlap = 15% of chunk_size)")
    for size in sizes:
        processor = DocumentProcessor(chunk_size=size)
        chunks = processor.process_document(text)
        test_chunk_token_bounds(processor, chunks)
        counts = [processor.calculate_tokens(c) for c in chunks]
        overlaps = [processor.calculate_tokens(shared_overlap(a, b)) for a, b in zip(chunks, chunks[1:])]
        print(f"size={size} overlap_cfg={processor.chunk_overlap} chunks={len(chunks)} "
              f"avg_tokens={sum(counts) / len(counts):.0f} max_tokens={max(counts)} "
              f"effective_overlaps={overlaps}")


if __name__ == "__main__":
    test_clean_text(DocumentProcessor())
    print()
    run_demo(SAMPLE_TEXT)
    run_size_sweep(build_long_text(SAMPLE_TEXT))
