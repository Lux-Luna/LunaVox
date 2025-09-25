import logging
from rich.logging import RichHandler
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# 以防万一写在导入库之前。
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

from .Client import Client

if __name__ == "__main__":
    cmd_client: Client = Client()
    cmd_client.run()
