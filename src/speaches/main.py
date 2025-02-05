from __future__ import annotations

import logging
import platform

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware

from speaches.dependencies import ApiKeyDependency, get_config
from speaches.logger import setup_logger
from speaches.routers.chat import (
    router as chat_router,
)
from speaches.routers.misc import (
    router as misc_router,
)
from speaches.routers.models import (
    router as models_router,
)
from speaches.routers.speech import (
    router as speech_router,
)
from speaches.routers.stt import (
    router as stt_router,
)
from speaches.routers.vad import (
    router as vad_router,
)
from speaches.routers.assistant import (
    router as assistant_router,
)


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    if platform.machine() != "x86_64":
        logger.warning(
            "`POST /v1/audio/speech` with `model=rhasspy/piper-voices` is only supported on x86_64 machines"
        )

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(
        dependencies=dependencies,
    )

    app.include_router(chat_router)
    app.include_router(stt_router)
    app.include_router(models_router)
    app.include_router(misc_router)
    app.include_router(speech_router)
    app.include_router(vad_router)
    app.include_router(assistant_router)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if config.enable_ui:
        import gradio as gr

        from speaches.ui.app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="/")

    return app
