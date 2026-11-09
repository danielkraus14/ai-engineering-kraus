"""Build the PDF deliverable from results.json (charts, flowchart, analysis, code)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).parent
DATA = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
OUT = HERE / "analisis_embeddings.pdf"
ACCENT = colors.HexColor("#1F3429")

IDS = list(DATA["relevant"]) + list(DATA["traps"])
TEXTS = {**DATA["relevant"], **DATA["traps"]}
RUN = DATA["runs"][0]


def heatmap() -> Path:
    path = HERE / "fig_heatmap.png"
    m = np.array(RUN["matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(m, cmap="YlGnBu", vmin=0.7, vmax=1.0)
    ax.set_xticks(range(len(IDS)), IDS)
    ax.set_yticks(range(len(IDS)), IDS)
    for i in range(len(IDS)):
        for j in range(len(IDS)):
            ax.text(j, i, f"{m[i][j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i][j] > 0.9 else "black")
    ax.axhline(4.5, color="red", lw=1.2)
    ax.axvline(4.5, color="red", lw=1.2)
    ax.set_title("Similitud coseno entre oraciones (embeddings)", fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def comparison_chart() -> Path:
    path = HERE / "fig_compare.png"
    x = np.arange(len(IDS))
    sem = [RUN["semantic"][i] for i in IDS]
    lex = [RUN["lexical"][i] for i in IDS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, vals, title, col in ((axes[0], sem, "Embeddings (coseno, eje truncado en 0,80)", "#4BAF9D"),
                                 (axes[1], lex, "Palabras clave (TF-IDF)", "#E28A3F")):
        bars = ax.bar(x, vals, color=[col if i.startswith("R") else "#999999" for i in IDS])
        ax.set_xticks(x, IDS)
        ax.set_title(title, fontsize=9)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.18)
    axes[0].set_ylim(0.8, 0.88)
    fig.suptitle('Query: "Despliegue de microservicios" (gris = trampas)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def flowchart() -> Path:
    path = HERE / "fig_flow.png"
    fig, ax = plt.subplots(figsize=(7.4, 8.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc="#D5E2DB", ec="#1F3429"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5)

    def arrow(p1, p2, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                     lw=1.3, color="#1F3429", linestyle=ls))

    # Online path (per query), left column
    box(3.2, 11.0, 4.4, 1.0, "INICIO\nEl usuario ingresa una query", fc="#E28A3F")
    box(3.2, 9.4, 4.4, 1.0, "Normalizar texto\n(limpiar espacios, truncar si es muy largo)")
    box(3.2, 7.8, 4.4, 1.0, "Modelo de embeddings\nquery  ->  vector q")
    box(3.2, 6.0, 4.4, 1.2, "Similitud coseno\ncos(q, d_i) contra el vector\nde cada documento")
    box(3.2, 4.2, 4.4, 1.0, "Ordenar scores de mayor a menor")
    box(3.2, 2.6, 4.4, 1.0, "Seleccionar el documento\ncon mayor score (top-1)")
    box(3.2, 1.0, 4.4, 1.0, "FIN\nDevolver documento + score", fc="#4BAF9D")
    for y1, y2 in ((10.5, 9.9), (8.9, 8.3), (7.3, 6.6), (5.4, 4.7), (3.7, 3.1), (2.1, 1.5)):
        arrow((3.2, y1), (3.2, y2))

    # Offline path (indexing), right column
    ax.text(7.9, 11.0, "Indexación (offline, una sola vez)", ha="center", fontsize=9,
            weight="bold", color="#1F3429")
    box(7.9, 9.4, 3.6, 1.0, "Corpus de documentos")
    box(7.9, 7.8, 3.6, 1.0, "Mismo modelo de embeddings\ndoc -> vector d_i")
    box(7.9, 6.0, 3.6, 1.2, "Índice / base vectorial\n(matriz de vectores d_i)", fc="#F3E3C8")
    arrow((7.9, 8.9), (7.9, 8.3))
    arrow((7.9, 7.3), (7.9, 6.6))
    arrow((6.1, 6.0), (5.4, 6.0))  # index -> cosine step
    ax.text(5.75, 6.35, "d_i", fontsize=8, ha="center")
    ax.text(7.9, 4.6, "Importante: query y documentos deben\nusar el MISMO modelo de embeddings,\n"
            "si no los vectores no son comparables.", ha="center", fontsize=8,
            style="italic", color="#7a2e0e")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], textColor=ACCENT, fontSize=20),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], textColor=ACCENT, fontSize=14, spaceBefore=12, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=ACCENT, fontSize=11.5, keepWithNext=1),
        "body": ParagraphStyle("b", parent=base["BodyText"], alignment=TA_JUSTIFY, leading=14, fontSize=10),
        "small": ParagraphStyle("s", parent=base["BodyText"], fontSize=8.5, leading=11),
        "code": ParagraphStyle("c", fontName="Courier", fontSize=7.6, leading=9.6,
                               backColor=colors.HexColor("#F2F5F3"), borderPadding=6),
    }


def table(rows, widths, header=True, font=8.5):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#3C4F44")),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F3")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def main() -> None:
    S = styles()
    P = lambda txt, st="body": Paragraph(txt, S[st])  # noqa: E731
    cell = lambda txt: Paragraph(txt, S["small"])  # noqa: E731
    m = np.array(RUN["matrix"])
    rel = [IDS.index(k) for k in DATA["relevant"]]
    trp = [IDS.index(k) for k in DATA["traps"]]
    intra = np.mean([m[i][j] for i in rel for j in rel if i < j])
    cross = np.mean([m[i][j] for i in rel for j in trp])
    sem, lex = RUN["semantic"], RUN["lexical"]

    el = []
    el += [P("Evaluación de un modelo de embeddings: relevancia semántica vs. coincidencia de palabras clave", "title"),
           P(f"Modelo evaluado: <b>{DATA['model']}</b> ({RUN['dims']} dimensiones) &nbsp;|&nbsp; "
             f"Concepto: <b>{DATA['concept']}</b> &nbsp;|&nbsp; Fecha de ejecución: {DATA['started'][:10]}", "small"),
           Spacer(1, 8)]

    el += [P("1. Objetivo y metodología", "h1"),
           P("Se evalúa si un modelo de embeddings distingue <i>significado</i> de <i>vocabulario compartido</i>. "
             "Se definen 5 oraciones que describen el mismo concepto técnico con vocabulario distinto (ninguna "
             "contiene las palabras \"despliegue\", \"microservicios\" ni \"servicio\") y 2 oraciones trampa que "
             "sí comparten palabras clave pero no significan lo mismo. Se mide la similitud coseno de cada oración "
             "contra el concepto y se la compara contra un baseline puramente léxico (TF-IDF). "
             "Toda la comparación se ejecutó 3 veces para comprobar que el resultado es estable.")]

    el += [P("2. Conceptos clave", "h1"),
           P("<b>Embedding:</b> vector numérico denso que representa el significado de un texto; textos con "
             "significado parecido quedan cerca en el espacio vectorial, aunque no compartan palabras."),
           Spacer(1, 4),
           P("<b>Similitud coseno:</b> mide el ángulo entre dos vectores: cos(a,b) = (a·b) / (‖a‖·‖b‖). Vale 1 si "
             "apuntan en la misma dirección, 0 si son ortogonales y -1 si son opuestos. Ignora la magnitud, por eso "
             "es la métrica estándar para comparar embeddings de textos de distinta longitud."),
           Spacer(1, 4),
           P("<b>Coincidencia léxica (TF-IDF):</b> representa el texto por las palabras que contiene. Si dos textos "
             "no comparten palabras, su similitud es 0 aunque hablen de lo mismo; si las comparten, es alta aunque "
             "el significado sea opuesto. Es el comportamiento que se quiere contrastar.")]

    el += [P("3. Conjunto de oraciones", "h1")]
    rows = [["ID", "Tipo", "Oración"]]
    for k, v in DATA["relevant"].items():
        rows.append([k, "Relevante (vocabulario distinto)", cell(v)])
    rows.append(["T1", "Trampa: irrelevante (homónimos)", cell(DATA["traps"]["T1"])])
    rows.append(["T2", "Trampa: significado opuesto", cell(DATA["traps"]["T2"])])
    el += [table(rows, [1.0 * cm, 4.2 * cm, 11.8 * cm]), Spacer(1, 4),
           P("T1 comparte \"servicio\", \"micro\" y \"despliegan\" pero habla de limpieza de oficinas. T2 comparte "
             "\"microservicios\" y \"despliegue\" pero rechaza la idea (postura opuesta).", "small")]

    el += [PageBreak(), P("4. Resultados", "h1"), P("4.1 Query corta contra cada oración", "h2")]
    rows = [["ID", "Coseno (embeddings)", "TF-IDF (palabras clave)", "Posición semántica"]]
    for pos, k in enumerate(RUN["ranking"], 1):
        rows.append([k, f"{sem[k]:.4f}", f"{lex[k]:.4f}", str(pos)])
    el += [table(rows, [2 * cm, 4.2 * cm, 4.6 * cm, 4 * cm]), Spacer(1, 6),
           Image(str(comparison_chart()), width=16.5 * cm, height=7.3 * cm), Spacer(1, 4),
           P("<b>Lectura.</b> El baseline léxico asigna 0,00 a dos oraciones relevantes (R3, R5) y su puntaje más "
             f"alto a la trampa T2 ({lex['T2']:.2f}): premia justamente las palabras repetidas. Los embeddings asignan "
             "puntajes altos a las 5 oraciones relevantes y colocan a T2 en último lugar. "
             "<b>Limitación observada:</b> con una query de solo 3 palabras los scores quedan comprimidos en un rango "
             f"muy estrecho ({min(sem.values()):.3f} a {max(sem.values()):.3f}) y R2 ({sem['R2']:.4f}) queda por debajo de la "
             f"trampa homónima T1 ({sem['T1']:.4f}), una diferencia de {sem['T1'] - sem['R2']:.4f}. "
             "La separación existe pero es frágil para queries tan cortas; el margen no alcanza para fijar un "
             "umbral de corte fiable.")]

    el += [P("4.2 Oración contra oración (más contexto)", "h2"),
           Image(str(heatmap()), width=10.8 * cm, height=9.1 * cm), Spacer(1, 4),
           P(f"Similitud promedio entre oraciones relevantes: <b>{intra:.3f}</b>. Similitud promedio entre una "
             f"relevante y una trampa: <b>{cross:.3f}</b>. La brecha ({intra - cross:.3f}) es mucho mayor que la "
             "diferencia observada con la query corta.")]

    loo = RUN["leave_one_out"]
    rows = [["Query", "Ranking de las demás oraciones (mayor a menor)", "Trampas al final"]]
    for q, ranked in loo.items():
        ok = set(ranked[-2:]) == set(DATA["traps"])
        rows.append([q, " > ".join(ranked), "Sí" if ok else "No"])
    el += [P("4.3 Leave-one-out: cada oración relevante como query", "h2"),
           table(rows, [2 * cm, 11 * cm, 4 * cm]), Spacer(1, 4),
           P("En los 5 casos las dos trampas quedan en las últimas posiciones: cuando la query tiene contexto "
             "suficiente, el modelo discrimina correctamente, incluso la trampa T2 que comparte las palabras exactas.")]

    el += [P("4.4 Verificación repetida", "h2")]
    rows = [["Corrida", "Ranking semántico (query corta)", "Diferencia máx. vs corrida 1"]]
    for r in DATA["runs"]:
        diff = max(abs(r["semantic"][k] - RUN["semantic"][k]) for k in IDS)
        rows.append([str(r["run"]), " > ".join(r["ranking"]), f"{diff:.2e}"])
    ck = DATA["checks"]
    el += [table(rows, [2 * cm, 10 * cm, 5 * cm]), Spacer(1, 4),
           P("Resumen de chequeos automáticos: "
             f"ranking idéntico entre corridas = <b>{'PASS' if ck['ranking_identical_across_runs'] else 'FAIL'}</b>; "
             f"leave-one-out con trampas al final = <b>{'PASS' if ck['leave_one_out_relevant_before_traps'] else 'FAIL'}</b>; "
             f"baseline léxico elige una trampa como primera = <b>{'PASS' if ck['keyword_baseline_ranks_a_trap_first'] else 'FAIL'}</b>; "
             f"query corta: todas las relevantes sobre T2 = <b>{'PASS' if ck['short_query_all_relevant_above_T2'] else 'FAIL'}</b>; "
             f"query corta: todas las relevantes sobre T1 = <b>{'PASS' if ck['short_query_all_relevant_above_T1'] else 'FAIL'}</b> "
             "(este último falla y se reporta tal cual).", "small")]

    el += [PageBreak(), P("5. Similitud coseno con scikit-learn", "h1"),
           P("<code>sklearn.metrics.pairwise.cosine_similarity(X, Y)</code> recibe arreglos 2D de forma "
             "(n_muestras, n_dimensiones) y devuelve una matriz con el coseno entre cada fila de X y cada fila de Y. "
             "Para dos vectores sueltos hay que darles forma (1, n) con <code>reshape(1, -1)</code> o listas anidadas. "
             "El código siguiente se ejecutó y produce los valores indicados en los comentarios."),
           Spacer(1, 4),
           Preformatted((HERE / "cosine_example.py").read_text(encoding="utf-8"), S["code"]),
           Spacer(1, 6),
           P("Cálculo aplicado al pipeline de este ejercicio (de <code>similarity.py</code>):", "body"),
           Preformatted(
               "def cosine_between(query, documents):\n"
               "    # query: (dims,)  documents: (n_docs, dims)  ->  scores: (n_docs,)\n"
               "    return cosine_similarity(query.reshape(1, -1), documents)[0]\n\n"
               "vectors = embed_texts([CONCEPT] + texts)      # Gemini, task_type=SEMANTIC_SIMILARITY\n"
               "semantic = cosine_between(vectors[0], vectors[1:])\n"
               "lexical = keyword_similarity(CONCEPT, texts)   # TF-IDF baseline\n", S["code"])]

    el += [PageBreak(), P("6. Diagrama de flujo: búsqueda semántica", "h1"),
           Image(str(flowchart()), width=14.6 * cm, height=16.6 * cm),
           P("El flujo de la izquierda se ejecuta en cada consulta; el de la derecha se ejecuta una vez (o cada vez "
             "que cambia el corpus) y deja los vectores listos para comparar. Implementación: "
             "<code>semantic_search.py</code>.", "small")]

    el += [P("7. Conclusiones", "h1"),
           P("1) El modelo agrupa correctamente oraciones con vocabulario totalmente distinto, y el baseline léxico "
             "demuestra por contraste que la palabra clave por sí sola engaña (puntúa 0 a paráfrasis válidas y más "
             "alto a la trampa de postura opuesta). <br/>2) Con una query corta de tres palabras, la ventaja de los "
             "embeddings existe pero es pequeña y una trampa homónima (T1) supera a una paráfrasis (R2) por " + f"{sem['T1'] - sem['R2']:.4f}".replace(".", ",") + ". "
             "<br/>3) Con oraciones completas como query, la separación es clara y estable. <br/>4) Recomendación "
             "de arquitectura: no depender de un umbral de coseno fijo con queries cortas; enriquecer la query, "
             "usar un reranker o combinar con búsqueda léxica (híbrida), y validar siempre con un conjunto "
             "de pruebas con trampas como este.")]

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Evaluación de embeddings", author="Daniel Kraus").build(el)
    print(f"PDF written to {OUT}")


if __name__ == "__main__":
    main()
