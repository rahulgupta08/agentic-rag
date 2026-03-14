import os

import pdfplumber
from pathlib import Path
from app.logging.logger import logger


class DocumentLoader:

    def __init__(self, file_path: str):

        self.file_path = file_path
        logger.info(f"Current directory in doc_loader class:={os.getcwd()}")
        logger.info(f"File path ::={self.file_path}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

    def load(self) -> dict:
        
        """
        Load PDF document and extract text page by page.
        """

        pages = []

        with pdfplumber.open(self.file_path) as pdf:

            for i, page in enumerate(pdf.pages):

                text = page.extract_text()

                if text:
                    text = self._normalize_text(text)

                    pages.append(
                        {
                            "page_number": i + 1,
                            "text": text
                        }
                    )

        logger.info(f"Loaded document:={self.file_path}")
        logger.info(f"Pages extracted: {len(pages)}")

        return {
            "file_path": self.file_path,
            "pages": pages
        }
    
    #  will return a structured document object like this:

    # {
    # "document_id": "apple_10k_2024",
    # "pages": [
    #     {"page_number": 1, "text": "..."},
    #     {"page_number": 2, "text": "..."}
    # ]
    # }

    # This allows us to keep page metadata for chunks later.

    def _normalize_text(self, text: str) -> str:
        """
        Clean extracted text.
        """

        text = text.replace("\n", " ")

        text = " ".join(text.split())

        return text