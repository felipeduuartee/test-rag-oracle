from dataclasses import dataclass
from pathlib import Path

from pypdfium2 import PdfDocument
from streamlit.runtime.uploaded_file_manager import UploadedFile

from config import Config

TEXT_FILE_EXTENSION = ".txt"
MD_FILE_EXTENSION = ".md"
PDF_EXTENSION = ".pdf"


@dataclass
class File:
    name: str
    content: str


def extract_pdf_content(data: bytes) -> str:
    pdf = PdfDocument(data)
    content = ""
    for page in pdf:
        text_page = page.get_textpage()
        #Extrai o texto da página
        content += f"{text_page.get_text_bounded()}\n"
    return content

def load_uploaded_file(uploaded_file: UploadedFile) -> File:
    file_extension = Path(uploaded_file.name).suffix

    #Verifica a extensão do arquivo pelo suffix
    if file_extension not in Config.ALLOWED_FILE_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension: {file_extension} for file {uploaded_file.name}"
        )

    #Se for pdf, usa a função para ler o pdf
    if file_extension == PDF_EXTENSION:
        return File(
            name=uploaded_file.name,
            content=extract_pdf_content(uploaded_file.getvalue())
        )

    #Caso seja texto
    return File(
        name=uploaded_file.name,
        content=uploaded_file.getvalue().decode("utf-8")
    )
