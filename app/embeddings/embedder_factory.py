from app.config import INGESTION_CONFIG

from .openai_embedder import OpenAIEmbedder
from .local_embedder import LocalEmbedder


def get_embedder(provider: str = None, **kwargs):
    """
    Returns an embedder instance.

    Args:
        provider (str): embedding provider ("local", "openai")
        kwargs: optional config params

    Returns:
        Embedder instance
    """

    # Backward compatibility (existing behavior)
    if provider is None:
        # Keep current default logic here
        return LocalEmbedder()

    if provider == "local":
        return LocalEmbedder(**kwargs)

    elif provider == "openai":
        return OpenAIEmbedder(**kwargs)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")