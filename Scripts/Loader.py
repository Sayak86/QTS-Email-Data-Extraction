from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    UnstructuredEmailLoader, OutlookMessageLoader,
    PyMuPDFLoader, UnstructuredPDFLoader,
    TesseractBlobParser
)
from langchain_core.documents import Document

def load_any(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    ocr = TesseractBlobParser()          # share across loaders

    for p in paths:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PyMuPDFLoader(
                str(p), mode="page",
                images_parser=ocr, images_inner_format="text"
            )
        elif suffix == ".msg":
            # keeps tables, screenshots, attachments
            loader = UnstructuredEmailLoader(
                str(p), mode="elements",
                process_attachments=True,
                images_parser=ocr,
                partition_kwargs={
                    "infer_table_structure": True,
                    "include_table_html": True,
                }
            )
        elif suffix == ".eml":
            loader = OutlookMessageLoader(str(p))  # fast text-only path
        else:
            print(f"Skipped unsupported file: {p.name}")
            continue

        docs.extend(loader.load())
    return docs

# ---- use it ----------------------------------------------------
files = list(Path("inbox_exports").iterdir())   # .msg, .eml, .pdf …
documents = load_any(files)

# concatenate page_content or pass the list straight to your next function
full_text = "\f".join(d.page_content for d in documents)
downstream_function(full_text)
