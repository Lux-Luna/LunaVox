from typing import TYPE_CHECKING, Optional
import logging

# Replace rich.console with standard logging
logger = logging.getLogger("LunaVox")

class ConsoleShim:
    """Shim for rich.console.Console to use standard logging."""
    def print(self, *args, **kwargs):
        # Join args with space if multiple arguments, similar to print
        msg = " ".join(str(arg) for arg in args)
        logger.info(msg)

console = ConsoleShim()
