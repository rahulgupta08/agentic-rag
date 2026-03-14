from app.bootstrap import bootstrap
bootstrap()
import logging
logger = logging.getLogger(__name__)

def log_node_start(node_name, state):
    logger.info(f"\nNODE START: {node_name}\nState keys: {list(state.keys()) if isinstance(state, dict) else state.model_dump().keys()}")



def log_node_end(node_name, state):
    logger.info(f"\n NODE END: {node_name}")