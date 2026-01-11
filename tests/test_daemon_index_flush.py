from pathlib import Path

import grepl.daemon.server as server


def _fake_reindex_factory(index, calls):
    def _fake_reindex():
        calls.append(set(index.dirty_files))
        index.dirty_files.clear()
        index._last_dirty_at = None

    return _fake_reindex


def test_flush_if_idle_reindexes(monkeypatch):
    index = server.InMemoryIndex(Path("/tmp/project"))
    calls = []

    monkeypatch.setattr(index, "_do_reindex", _fake_reindex_factory(index, calls))

    now = 1000.0

    def fake_time():
        return now

    monkeypatch.setattr(server.time, "time", fake_time)

    index.mark_dirty("/tmp/project/file_a.py")
    assert calls == []

    now += server.DIRTY_IDLE_FLUSH_SECONDS - 0.1
    index.flush_if_idle(server.DIRTY_IDLE_FLUSH_SECONDS)
    assert calls == []

    now += 0.2
    index.flush_if_idle(server.DIRTY_IDLE_FLUSH_SECONDS)
    assert calls == [{"/tmp/project/file_a.py"}]


def test_flush_reindexes_pending(monkeypatch):
    index = server.InMemoryIndex(Path("/tmp/project"))
    calls = []

    monkeypatch.setattr(index, "_do_reindex", _fake_reindex_factory(index, calls))

    index.dirty_files.add("/tmp/project/file_b.py")
    index.flush()
    assert calls == [{"/tmp/project/file_b.py"}]
