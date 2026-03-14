import sys
sys.dont_write_bytecode = True
from app.logging.logger import setup_logging

def bootstrap():
    """
    Application initialization.
    Called before any script execution.
    """
    setup_logging()