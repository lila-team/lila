import logging


# Create a passthrough handler to avoid browser-use
# to setup its own logging configuration
# https://github.com/browser-use/browser-use/blob/5e0af71af83e0da594044e23758c9564f91a185f/browser_use/logging_config.py#L71
# Lila uses loguru.
class PassHandler(logging.Handler):
    def emit(self, record):
        pass


# Add a passthrough handler to the root logger
logging.getLogger().addHandler(PassHandler())
