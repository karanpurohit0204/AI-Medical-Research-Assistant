import sys
from loguru import logger
from backend.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()

    if settings.is_production:
        logger.add(
            sys.stdout,
            level="INFO",
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stdout,
            level="DEBUG",
            colorize=True,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logging initialised | env={settings.app_env}")


__all__ = ["logger", "setup_logging"]