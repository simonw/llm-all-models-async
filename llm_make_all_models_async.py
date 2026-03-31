import asyncio
import llm
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional


_executor = ThreadPoolExecutor(max_workers=4)


class AsyncModelWrapper(llm.AsyncModel):
    """Wraps a sync Model to make it available as an AsyncModel using a thread pool."""

    def __init__(self, sync_model):
        self._sync_model = sync_model
        self.model_id = sync_model.model_id
        self.can_stream = sync_model.can_stream
        self.attachment_types = sync_model.attachment_types
        self.supports_schema = sync_model.supports_schema
        self.supports_tools = sync_model.supports_tools

    def __str__(self):
        return str(self._sync_model)

    async def execute(
        self,
        prompt,
        stream,
        response,
        conversation,
    ) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_event_loop()

        def _run_sync():
            try:
                for chunk in self._sync_model.execute(
                    prompt, stream, response, conversation
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(_executor, _run_sync)

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item


class AsyncKeyModelWrapper(llm.AsyncKeyModel):
    """Wraps a sync KeyModel to make it available as an AsyncKeyModel."""

    def __init__(self, sync_model):
        self._sync_model = sync_model
        self.model_id = sync_model.model_id
        self.can_stream = sync_model.can_stream
        self.attachment_types = sync_model.attachment_types
        self.supports_schema = sync_model.supports_schema
        self.supports_tools = sync_model.supports_tools
        self.needs_key = sync_model.needs_key
        self.key_env_var = sync_model.key_env_var

    def __str__(self):
        return str(self._sync_model)

    async def execute(
        self,
        prompt,
        stream,
        response,
        conversation,
        key=None,
    ) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_event_loop()

        def _run_sync():
            try:
                for chunk in self._sync_model.execute(
                    prompt, stream, response, conversation, key=key
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(_executor, _run_sync)

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item


def _make_async_wrapper(sync_model):
    """Create an async wrapper for a sync model."""
    if isinstance(sync_model, llm.KeyModel):
        return AsyncKeyModelWrapper(sync_model)
    return AsyncModelWrapper(sync_model)


# Monkey-patch ModelWithAliases.__init__ to automatically add async wrappers
# for sync-only models. This is the most reliable approach because
# ModelWithAliases instances are created inside the register() callback,
# which runs AFTER load_plugins() completes. So even on the very first call
# to get_models_with_aliases(), all models will be wrapped.
_original_mwa_init = llm.ModelWithAliases.__init__


def _patched_mwa_init(self, model, async_model, aliases):
    if model and not async_model:
        async_model = _make_async_wrapper(model)
    _original_mwa_init(self, model, async_model, aliases)


llm.ModelWithAliases.__init__ = _patched_mwa_init
