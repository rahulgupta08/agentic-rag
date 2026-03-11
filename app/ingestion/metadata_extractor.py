import os


TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta"
}


def extract_metadata_from_filename(file_path: str):

    filename = os.path.basename(file_path)

    name = filename.replace(".pdf", "")

    # Example: aapl-10K_2023
    parts = name.split("_")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid filename format: {filename}"
        )

    ticker_doc = parts[0]
    year = parts[1]

    ticker, document_type = ticker_doc.split("-")

    ticker = ticker.upper()

    metadata = {
        "ticker": ticker,
        "company": TICKER_TO_COMPANY.get(ticker, ticker),
        "year": year,
        "document_type": document_type.lower()
    }

    return metadata