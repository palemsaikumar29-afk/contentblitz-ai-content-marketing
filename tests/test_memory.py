"""Conversation memory tests."""
from contentblitz.memory import ConversationMemory


def test_add_and_history():
    mem = ConversationMemory()
    mem.add_turn("user", "write a blog about churn")
    mem.add_turn("assistant", "here is a draft")
    assert len(mem) == 2
    hist = mem.history()
    assert hist[0] == {"role": "user", "content": "write a blog about churn"}


def test_history_n_limits():
    mem = ConversationMemory()
    for i in range(5):
        mem.add_turn("user", f"msg {i}")
    assert len(mem.history(2)) == 2
    assert mem.history(2)[-1]["content"] == "msg 4"


def test_history_text_compact():
    mem = ConversationMemory()
    mem.add_turn("user", "hello")
    text = mem.history_text()
    assert "USER: hello" in text


def test_brief_update_and_outputs():
    mem = ConversationMemory()
    mem.update_brief(topic="churn", audience=None)
    assert mem.brief == {"topic": "churn"}
    mem.remember_output("blog", "draft text")
    assert mem.last_output("blog") == "draft text"
    assert mem.last_output("linkedin") is None


def test_research_memory():
    mem = ConversationMemory()
    mem.remember_research([{"title": "t"}])
    assert mem.last_research == [{"title": "t"}]


def test_clear():
    mem = ConversationMemory()
    mem.add_turn("user", "hi")
    mem.update_brief(topic="x")
    mem.clear()
    assert len(mem) == 0
    assert mem.brief == {}
