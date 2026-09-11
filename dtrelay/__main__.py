"""Run the relay: python3 -m dtrelay"""
import logging

import uvicorn

from dtrelay.config import load_settings
from dtrelay.server import create_app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.port)


if __name__ == "__main__":
    main()
