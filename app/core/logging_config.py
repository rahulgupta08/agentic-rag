import logging
import sys
import os

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log')
print("cwd:", os.getcwd())
expected_logfile = os.path.join(os.getcwd(), 'app.log')
print("expected logfile:", expected_logfile)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        filename='app.log'
       # force=True
    )