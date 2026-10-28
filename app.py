from __future__ import annotations

from io import BytesIO
import streamlit as st
from docx import Document
from google import genai
from pypdf import PdfReader

MODEL_NAME = "gemini-3.8-flash"
MAX_NOTE_CHARACTERS = 60_000
SUPPORTED_EXTENSIONS = {"pdf", "txt", "docx"}

st.set_page_config(page_title="Study Notes Assistant", page_icon="📚", layout="wide")
st.title("📚 Study Notes Assistant")
st.write("Turn your notes into a summary, flashcards, or a multiple-choice quiz.")


def extract_text(uploaded_file) -> str:
    """Extract readable text from a supported Streamlit upload."""
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError(f"{uploaded_file.name} is empty.")

    if extension == "txt":
        try:
            return data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{uploaded_file.name} is not valid UTF-8 text.") from exc
    if extension == "pdf":
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if extension == "docx":
        document = Document(BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        # Include table contents, which often contain class notes or study guides.
        for table in document.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(paragraphs).strip()
    raise ValueError(f"{uploaded_file.name} has an unsupported file type. Use PDF, TXT, or DOCX.")


def build_prompt(notes: str, output_type: str, instructions: str) -> str:
    """Create a focused prompt that asks the model to use the supplied notes."""
    formats = {
        "Summary": "Write a concise summary with clear headings and a bullet list of key concepts. Preserve important definitions and relationships.",
        "Flashcards": "Create useful study flashcards in question-and-answer format. Number each card and clearly label Question and Answer.",
        "Multiple-choice quiz": "Create a multiple-choice quiz. Give each question four labeled options, then provide an answer key with the correct option and a short explanation for each answer.",
    }
    extra = instructions.strip() or "No additional topic or instructions were provided. Cover the most important material in the notes."
    return f"""You are a careful study assistant. Generate the requested study material using the notes below as the source of truth. Do not invent facts absent from the notes. If the notes do not support an answer, say so.

Requested material: {output_type}
Task: {formats[output_type]}
User topic or instructions: {extra}

Notes begin:
{notes}
Notes end.

Use readable Markdown headings and concise language. Do not claim the material is guaranteed correct."""


def generate_material(notes: str, output_type: str, instructions: str) -> str:
    """Call Gemini Flash with the key stored in Streamlit secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError("Gemini API key is not configured. Add GEMINI_API_KEY to .streamlit/secrets.toml or the Streamlit Cloud app secrets.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is empty. Add a valid key to your Streamlit secrets.")

    client = genai.Client(api_key=api_key.strip())
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_prompt(notes, output_type, instructions),
    )
    result = response.text
    if not result or not result.strip():
        raise RuntimeError("Gemini returned no readable text. Try again or use shorter notes.")
    return result.strip()


uploaded_files = st.file_uploader(
    "Upload study notes (PDF, TXT, or DOCX)",
    type=sorted(SUPPORTED_EXTENSIONS),
    accept_multiple_files=True,
    help="Scanned PDFs without a text layer may not yield extractable text.",
)
output_type = st.selectbox("What would you like to create?", ["Summary", "Flashcards", "Multiple-choice quiz"])
instructions = st.text_area("Topic or specific instructions (optional)", placeholder="For example: Focus on photosynthesis and include 8 flashcards.")

if uploaded_files:
    extracted_parts: list[str] = []
    valid_names: list[str] = []
    for uploaded_file in uploaded_files:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
        if extension not in SUPPORTED_EXTENSIONS:
            st.error(f"{uploaded_file.name}: unsupported file type. Use PDF, TXT, or DOCX.")
            continue
        try:
            text = extract_text(uploaded_file)
            if not text.strip():
                st.warning(f"{uploaded_file.name}: no extractable text was found. Scanned/image-only PDFs need OCR before upload.")
                continue
            extracted_parts.append(f"--- {uploaded_file.name} ---\n{text}")
            valid_names.append(uploaded_file.name)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error(f"Could not read {uploaded_file.name}. Check that it is a valid, unencrypted {extension.upper()} file.")

    notes = "\n\n".join(extracted_parts)
    if notes:
        if len(notes) > MAX_NOTE_CHARACTERS:
            notes = notes[:MAX_NOTE_CHARACTERS]
            st.warning(f"The combined extracted notes exceeded {MAX_NOTE_CHARACTERS:,} characters. Only the first {MAX_NOTE_CHARACTERS:,} characters will be sent to Gemini; later content may be omitted.")
        else:
            st.caption(f"Ready to use {len(valid_names)} file(s), {len(notes):,} characters total.")

        if st.button("Generate study material", type="primary"):
            with st.spinner("Generating your study material…"):
                try:
                    generated = generate_material(notes, output_type, instructions)
                    st.session_state["generated"] = generated
                    st.session_state["generated_type"] = output_type
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception:
                    # Avoid showing SDK exception details that could contain request metadata.
                    st.error("Gemini could not complete the request. Check your API key, connection, and quota, then try again with shorter notes.")

if st.session_state.get("generated"):
    st.divider()
    st.subheader(st.session_state.get("generated_type", "Generated study material"))
    st.markdown(st.session_state["generated"])
    st.download_button(
        "Download as .txt",
        data=st.session_state["generated"],
        file_name="study_material.txt",
        mime="text/plain; charset=utf-8",
    )
    st.caption("AI-generated study aid: verify important details and answers against your course sources.")
