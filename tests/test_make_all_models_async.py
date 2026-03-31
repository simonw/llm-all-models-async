import asyncio
import llm
import llm_make_all_models_async  # noqa: F401 - triggers monkey-patching
from llm_make_all_models_async import AsyncModelWrapper, AsyncKeyModelWrapper
import pytest


class SyncOnlyModel(llm.Model):
    model_id = "sync-only-test"
    can_stream = True

    def execute(self, prompt, stream, response, conversation):
        yield "Hello "
        yield "World"


@pytest.fixture
def register_sync_only():
    class TestPlugin:
        __name__ = "test_sync_only"

        @llm.hookimpl
        def register_models(self, register):
            register(SyncOnlyModel())

    llm.pm.register(TestPlugin(), name="test_sync_only")
    yield
    llm.pm.unregister(name="test_sync_only")


def test_sync_model_gets_async_wrapper(register_sync_only):
    """A sync-only model should automatically get an async counterpart."""
    async_model = llm.get_async_model("sync-only-test")
    assert async_model is not None
    assert async_model.model_id == "sync-only-test"
    assert isinstance(async_model, AsyncModelWrapper)


def test_async_wrapper_produces_correct_output(register_sync_only):
    """The async wrapper should produce the same text as the sync model."""
    async_model = llm.get_async_model("sync-only-test")
    response = async_model.prompt("hi")
    result = asyncio.run(response.text())
    assert result == "Hello World"


async def test_async_streaming(register_sync_only):
    """The async wrapper should support streaming via async iteration."""
    async_model = llm.get_async_model("sync-only-test")
    response = async_model.prompt("hi")
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    assert chunks == ["Hello ", "World"]


def test_already_async_model_not_wrapped():
    """Models that already have an async variant should not be double-wrapped."""

    class ExistingAsync(llm.AsyncModel):
        model_id = "has-async"

        async def execute(self, prompt, stream, response, conversation):
            yield "async original"

    class TestPlugin:
        __name__ = "test_no_double_wrap"

        @llm.hookimpl
        def register_models(self, register):
            register(SyncOnlyModel(), ExistingAsync())

    llm.pm.register(TestPlugin(), name="test_no_double_wrap")
    try:
        async_model = llm.get_async_model("sync-only-test")
        assert isinstance(async_model, ExistingAsync)
    finally:
        llm.pm.unregister(name="test_no_double_wrap")


def test_key_model_gets_async_key_wrapper():
    """A sync KeyModel should get an AsyncKeyModel wrapper."""

    class SyncKeyModel(llm.KeyModel):
        model_id = "sync-key-test"
        needs_key = "test"
        key_env_var = "TEST_KEY"

        def execute(self, prompt, stream, response, conversation, key=None):
            yield f"key={key}"

    class TestPlugin:
        __name__ = "test_key_model"

        @llm.hookimpl
        def register_models(self, register):
            register(SyncKeyModel())

    llm.pm.register(TestPlugin(), name="test_key_model")
    try:
        async_model = llm.get_async_model("sync-key-test")
        assert isinstance(async_model, AsyncKeyModelWrapper)
        assert async_model.needs_key == "test"
        assert async_model.key_env_var == "TEST_KEY"
    finally:
        llm.pm.unregister(name="test_key_model")


def test_get_async_models_includes_wrapped(register_sync_only):
    """llm.get_async_models() should include wrapped sync-only models."""
    async_models = llm.get_async_models()
    model_ids = [m.model_id for m in async_models]
    assert "sync-only-test" in model_ids


def test_get_async_model_aliases_includes_wrapped():
    """llm.get_async_model_aliases() should include wrapped sync-only models."""

    class AliasedSyncModel(llm.Model):
        model_id = "aliased-sync-test"
        can_stream = True

        def execute(self, prompt, stream, response, conversation):
            yield "hi"

    class TestPlugin:
        __name__ = "test_aliased"

        @llm.hookimpl
        def register_models(self, register):
            register(AliasedSyncModel(), aliases=("aliased-test",))

    llm.pm.register(TestPlugin(), name="test_aliased")
    try:
        aliases = llm.get_async_model_aliases()
        assert "aliased-sync-test" in aliases
        assert "aliased-test" in aliases
    finally:
        llm.pm.unregister(name="test_aliased")


def test_sync_model_error_propagates(register_sync_only):
    """Errors from the sync model should propagate through the async wrapper."""

    class ErrorModel(llm.Model):
        model_id = "error-test"

        def execute(self, prompt, stream, response, conversation):
            raise ValueError("test error")
            yield  # make it a generator

    class TestPlugin:
        __name__ = "test_error"

        @llm.hookimpl
        def register_models(self, register):
            register(ErrorModel())

    llm.pm.register(TestPlugin(), name="test_error")
    try:
        async_model = llm.get_async_model("error-test")
        response = async_model.prompt("hi")
        with pytest.raises(ValueError, match="test error"):
            asyncio.run(response.text())
    finally:
        llm.pm.unregister(name="test_error")
