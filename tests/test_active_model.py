import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_active_model_uses_existing_openai_compatible_client():
    from scripts.core.active_model import call_active_model

    class Message:
        content = '[{"prompt":"new","answer_outline":"answer"}]'
    class Choice:
        message = Message()
    class Response:
        choices = [Choice()]
    class Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "oauth-model"
            return Response()
    class Agent:
        model = "oauth-model"
        class Client:
            chat = type("Chat", (), {"completions": Completions()})()
        client = Client()

    assert "new" in call_active_model(Agent(), "make a quiz")


def test_active_model_fails_explicitly_without_client():
    from scripts.core.active_model import call_active_model
    try:
        call_active_model(object(), "make a quiz")
    except RuntimeError as exc:
        assert "does not expose" in str(exc)
    else:
        raise AssertionError("expected explicit transport error")
