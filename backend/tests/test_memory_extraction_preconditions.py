"""Extraction proposals retain the source state observed by the model."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from deerflow.agents.memory.backends.deermem.deermem.config import DeerMemConfig
from deerflow.agents.memory.backends.deermem.deermem.core.storage import FileMemoryStorage
from deerflow.agents.memory.backends.deermem.deermem.core.updater import MemoryUpdater

SCOPE = {"user_id": "alice", "agent_name": "researcher"}
THREAD = "extraction"
OLD = "The user works on distributed systems."
PEER = "The user works on accessibility."
PROPOSED = "The user works on distributed systems and prefers concise reports."
MESSAGE = HumanMessage(content="I prefer concise reports and withdraw my old preference.", id="new-turn")


class Model:
    def __init__(self, payload, during_invoke=lambda: None):
        self.payload = payload
        self.during_invoke = during_invoke
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append("\n".join(str(message.content) for message in prompt))
        self.during_invoke()
        return SimpleNamespace(content=json.dumps({"user": {}, "history": {}, "newFacts": [], **self.payload}))


def runtime(path, model=None, **overrides):
    config = DeerMemConfig(**{"storage_path": str(path), "staleness_review_enabled": False, "token_counting": "char", **overrides})
    storage = FileMemoryStorage(config)
    return storage, MemoryUpdater(config, storage, model)


def summary_payload(group="user", field="workContext", text=PROPOSED):
    return {group: {field: {"shouldUpdate": True, "summary": text, "scope": "user", "authority": "descriptive"}}}


def write_summary(path, *, group="user", field="workContext", text=PEER, scope=None):
    _, updater = runtime(path, Model(summary_payload(group, field, text)))
    assert updater.update_memory([MESSAGE], **(scope or SCOPE))


def extract(updater, *, asynchronous=False, messages=None, **kwargs):
    method = updater.aupdate_memory if asynchronous else updater.update_memory
    result = method(messages or [MESSAGE], thread_id=THREAD, **SCOPE, **kwargs)
    return asyncio.run(result) if asynchronous else result


def fresh(path):
    storage, _ = runtime(path)
    return storage.load(**SCOPE)


def assert_retryable(path, updater, model, *, asynchronous=False, prior=None):
    storage, _ = runtime(path)
    assert storage.get_extraction_watermark(THREAD, **SCOPE) == prior
    # Both the in-process LRU and a restarted worker must retain the failed feed.
    model.during_invoke = lambda: None
    model.payload = {}
    assert extract(updater, asynchronous=asynchronous)
    assert len(model.prompts) == 2
    assert storage.get_extraction_watermark(THREAD, **SCOPE) == ["id", MESSAGE.id]
    restarted = Model({})
    _, new_updater = runtime(path, restarted)
    assert extract(new_updater, asynchronous=asynchronous)
    assert restarted.prompts == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("group", "field"),
    [("user", "workContext"), ("user", "personalContext"), ("user", "topOfMind"), ("user", "cognitiveStyle"), ("history", "recentMonths"), ("history", "earlierContext"), ("history", "longTermBackground")],
)
def test_same_summary_field_conflict_keeps_peer_and_watermark(tmp_path, asynchronous, group, field):
    write_summary(tmp_path, group=group, field=field, text=OLD)
    model = Model(summary_payload(group, field), lambda: write_summary(tmp_path, group=group, field=field))
    _, updater = runtime(tmp_path, model)

    assert extract(updater, asynchronous=asynchronous) is False
    assert OLD in model.prompts[0]
    assert fresh(tmp_path)[group][field]["summary"] == PEER
    assert_retryable(tmp_path, updater, model, asynchronous=asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("with_summary", [False, True])
@pytest.mark.parametrize("operation", ["correct", "delete"])
def test_removal_keeps_original_fact_revision(tmp_path, asynchronous, with_summary, operation):
    peer_storage, peer = runtime(tmp_path)
    _, fact_id = peer.create_memory_fact("The user prefers weekly reports.", confidence=0.9, **SCOPE)
    payload = summary_payload() if with_summary else {}
    payload["factsToRemove"] = [{"id": fact_id, "scope": "user", "reason": "User withdrew the reporting preference."}]

    def concurrent_write():
        if operation == "correct":
            peer.update_memory_fact(fact_id, content="The user prefers daily reports.", **SCOPE)
        else:
            peer.delete_memory_fact(fact_id, **SCOPE)

    model = Model(payload, concurrent_write)
    _, updater = runtime(tmp_path, model)
    assert extract(updater, asynchronous=asynchronous) is False
    assert "weekly reports" in model.prompts[0]
    data = fresh(tmp_path)
    if operation == "correct":
        assert data["facts"] == peer_storage.load(**SCOPE)["facts"]
        assert data["facts"][0]["content"] == "The user prefers daily reports."
        assert data["facts"][0]["revision"] == 2
    else:
        assert data["facts"] == []
    assert data["user"]["workContext"]["summary"] == ""
    assert_retryable(tmp_path, updater, model, asynchronous=asynchronous)


@pytest.mark.parametrize("peer_change", ["none", "other_field", "other_user", "other_agent", "other_fact", "read"])
def test_unrelated_changes_remain_mergeable(tmp_path, peer_change):
    write_summary(tmp_path, text=OLD)
    _, peer = runtime(tmp_path)
    _, fact_id = peer.create_memory_fact("The user prefers weekly reports.", **SCOPE)
    _, other_id = peer.create_memory_fact("The user likes source links.", **SCOPE)
    payload = summary_payload()
    payload["factsToRemove"] = [{"id": fact_id, "scope": "user", "reason": "User withdrew the reporting preference."}]

    def concurrent_write():
        if peer_change == "other_field":
            write_summary(tmp_path, field="topOfMind")
        elif peer_change == "other_user":
            write_summary(tmp_path, scope={**SCOPE, "user_id": "bob"})
        elif peer_change == "other_agent":
            peer.create_memory_fact("The user likes detailed reviews.", **{**SCOPE, "agent_name": "reviewer"})
        elif peer_change == "other_fact":
            peer.update_memory_fact(other_id, content="The user likes primary source links.", **SCOPE)
        elif peer_change == "read":
            peer.get_memory_data(**SCOPE)

    model = Model(payload, concurrent_write)
    _, updater = runtime(tmp_path, model)
    assert extract(updater)
    data = fresh(tmp_path)
    assert data["user"]["workContext"]["summary"] == PROPOSED
    assert [fact["id"] for fact in data["facts"]] == [other_id]
    if peer_change == "other_field":
        assert data["user"]["topOfMind"]["summary"] == PEER
    if peer_change == "other_fact":
        assert data["facts"][0]["content"] == "The user likes primary source links."
    assert len(model.prompts) == 1


def test_shared_summary_conflicts_across_agents(tmp_path):
    write_summary(tmp_path, text=OLD)
    model = Model(summary_payload(), lambda: write_summary(tmp_path, scope={**SCOPE, "agent_name": "reviewer"}))
    _, updater = runtime(tmp_path, model)
    assert extract(updater) is False
    assert fresh(tmp_path)["user"]["workContext"]["summary"] == PEER


@pytest.mark.parametrize("conflicts", [1, 2, 3])
def test_repeated_manifest_conflicts_use_bounded_budget(tmp_path, monkeypatch, conflicts):
    write_summary(tmp_path, text=OLD)
    model = Model(summary_payload())
    storage, updater = runtime(tmp_path, model)
    original_apply = storage.apply_changes
    attempts = []

    def interleaved_apply(change_set, **kwargs):
        attempts.append(copy.deepcopy(change_set))
        if len(attempts) <= conflicts:
            write_summary(tmp_path, field="topOfMind", text=f"Peer revision {len(attempts)}.")
        return original_apply(change_set, **kwargs)

    monkeypatch.setattr(storage, "apply_changes", interleaved_apply)
    assert extract(updater) is (conflicts < 3)
    assert len(attempts) == min(conflicts + 1, 3)
    assert fresh(tmp_path)["user"]["topOfMind"]["summary"] == f"Peer revision {conflicts}."
    if conflicts == 3:
        assert fresh(tmp_path)["user"]["workContext"]["summary"] == OLD
        assert_retryable(tmp_path, updater, model)


def test_original_precondition_survives_an_unrelated_retry(tmp_path, monkeypatch):
    write_summary(tmp_path, text=OLD)
    storage, updater = runtime(tmp_path, Model(summary_payload()))
    original_apply = storage.apply_changes
    attempts = []

    def interleaved_apply(change_set, **kwargs):
        attempts.append(change_set)
        write_summary(tmp_path, field="topOfMind" if len(attempts) == 1 else "workContext")
        return original_apply(change_set, **kwargs)

    monkeypatch.setattr(storage, "apply_changes", interleaved_apply)
    assert extract(updater) is False
    assert len(attempts) == 2
    assert fresh(tmp_path)["user"]["workContext"]["summary"] == PEER
    assert storage.get_extraction_watermark(THREAD, **SCOPE) is None


def test_storage_failure_preserves_previous_watermark(tmp_path, monkeypatch):
    model = Model({})
    storage, updater = runtime(tmp_path, model)
    previous = HumanMessage(content="A prior completed turn.", id="prior-turn")
    assert extract(updater, messages=[previous])
    model.prompts.clear()
    model.payload = summary_payload()
    before = fresh(tmp_path)
    original_commit = storage._commit_changes_locked

    def fail_commit(*args, **kwargs):
        raise OSError("Injected storage failure")

    with monkeypatch.context() as failure:
        failure.setattr(storage, "_commit_changes_locked", fail_commit)
        assert extract(updater, messages=[previous, MESSAGE]) is False
    assert fresh(tmp_path) == before
    assert storage._commit_changes_locked == original_commit
    assert_retryable(tmp_path, updater, model, prior=["id", previous.id])


def test_restarted_worker_reextracts_conflict_from_new_source(tmp_path):
    write_summary(tmp_path, text=OLD)
    previous = HumanMessage(content="A completed turn.", id="prior-turn")
    storage, completed = runtime(tmp_path, Model({}))
    assert extract(completed, messages=[previous])
    late = Model(summary_payload(), lambda: write_summary(tmp_path))
    _, updater = runtime(tmp_path, late)
    assert extract(updater, messages=[previous, MESSAGE]) is False
    assert storage.get_extraction_watermark(THREAD, **SCOPE) == ["id", previous.id]

    reextracted = PEER + " The user prefers concise reports."
    model = Model(summary_payload(text=reextracted))
    _, restarted = runtime(tmp_path, model)
    assert extract(restarted, messages=[previous, MESSAGE])
    assert len(model.prompts) == 1
    assert PEER in model.prompts[0]
    assert previous.content not in model.prompts[0]
    assert fresh(tmp_path)["user"]["workContext"]["summary"] == reextracted
    assert storage.get_extraction_watermark(THREAD, **SCOPE) == ["id", MESSAGE.id]


def test_fact_revision_detects_edit_and_restore(tmp_path):
    _, peer = runtime(tmp_path)
    content = "The user prefers weekly reports."
    _, fact_id = peer.create_memory_fact(content, **SCOPE)

    def edit_and_restore():
        peer.update_memory_fact(fact_id, content="The user prefers daily reports.", **SCOPE)
        peer.update_memory_fact(fact_id, content=content, **SCOPE)

    model = Model({"factsToRemove": [{"id": fact_id, "scope": "user", "reason": "User withdrew the old preference."}]}, edit_and_restore)
    _, updater = runtime(tmp_path, model)
    assert extract(updater) is False
    assert fresh(tmp_path)["facts"][0]["revision"] == 3
    assert fresh(tmp_path)["facts"][0]["content"] == content


@pytest.mark.parametrize("kind", ["summary", "removal"])
def test_rejected_proposal_does_not_block_unrelated_updates(tmp_path, kind):
    write_summary(tmp_path, text=OLD)
    _, peer = runtime(tmp_path)
    _, fact_id = peer.create_memory_fact("The user prefers weekly reports.", **SCOPE)
    payload = summary_payload(field="topOfMind")
    if kind == "summary":
        payload["user"]["workContext"] = {"shouldUpdate": True, "summary": PROPOSED, "scope": "thread", "authority": "descriptive"}

        def callback():
            write_summary(tmp_path)
    else:
        payload["factsToRemove"] = [{"id": fact_id, "scope": "thread", "reason": "Task-local proposal."}]

        def callback():
            peer.update_memory_fact(fact_id, content="The user prefers daily reports.", **SCOPE)

    _, updater = runtime(tmp_path, Model(payload, callback))
    assert extract(updater)
    assert fresh(tmp_path)["user"]["topOfMind"]["summary"] == PROPOSED
    if kind == "summary":
        assert fresh(tmp_path)["user"]["workContext"]["summary"] == PEER
    else:
        assert fresh(tmp_path)["facts"][0]["content"] == "The user prefers daily reports."


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("operation", ["correct", "delete", "import_summary"])
def test_http_management_write_survives_inflight_extraction(tmp_path, monkeypatch, asynchronous, operation):
    from _router_auth_helpers import make_authed_test_app
    from fastapi.testclient import TestClient

    from app.gateway.auth.models import User
    from app.gateway.routers import memory
    from deerflow.agents.memory.backends.deermem.deer_mem import DeerMem

    user = User(email="memory-test@example.com", password_hash="x", system_role="user")
    scope = {**SCOPE, "user_id": str(user.id)}
    app = make_authed_test_app(user_factory=lambda: user, bind_current_user=True)
    app.include_router(memory.router)
    manager = DeerMem(backend_config={"storage_path": str(tmp_path)})
    monkeypatch.setattr(memory, "get_memory_manager", lambda: manager)
    _, peer = runtime(tmp_path)
    _, fact_id = peer.create_memory_fact("The user prefers weekly reports.", **scope)
    write_summary(tmp_path, text=OLD, scope=scope)
    payload = summary_payload()
    payload["factsToRemove"] = [{"id": fact_id, "scope": "user", "reason": "User withdrew the old preference."}]
    committed = []
    with TestClient(app) as client:

        def management_write():
            url = f"/api/memory/facts/{fact_id}?agent_name=researcher"
            if operation == "correct":
                response = client.patch(url, json={"content": "The user prefers daily reports."})
            elif operation == "delete":
                response = client.delete(url)
            else:
                document = client.get("/api/memory/export").json()
                document["user"]["workContext"]["summary"] = PEER
                response = client.post("/api/memory/import", json=document)
            assert response.status_code == 200, response.text
            committed.append(peer.reload_memory_data(**scope))

        model = Model(payload, management_write)
        storage, updater = runtime(tmp_path, model)
        method = updater.aupdate_memory if asynchronous else updater.update_memory
        result = method([MESSAGE], thread_id=THREAD, **scope)
        assert (asyncio.run(result) if asynchronous else result) is False
        assert len(committed) == 1
        assert peer.reload_memory_data(**scope) == committed[0]
        assert storage.get_extraction_watermark(THREAD, **scope) is None


def test_completed_extraction_allows_later_management_write(tmp_path):
    model = Model(summary_payload())
    _, updater = runtime(tmp_path, model)
    assert extract(updater)
    write_summary(tmp_path)
    assert fresh(tmp_path)["user"]["workContext"]["summary"] == PEER


@pytest.mark.parametrize("operation", ["stale_remove", "stale_extend", "consolidate", "reinforce"])
def test_other_fact_mutations_preserve_source_revision(tmp_path, operation):
    options = {"consolidation_enabled": True, "consolidation_min_facts": 3, "fact_eviction_policy": "hybrid-v1", "staleness_review_enabled": True}
    storage, peer = runtime(tmp_path, **options)
    ids = [f"fact_format_{index}" for index in range(3)]
    data = storage.load(**SCOPE)
    data["facts"] = [{"id": fact_id, "content": f"The user prefers report format {index}.", "category": "context", "confidence": 0.9, "createdAt": "2000-01-01T00:00:00Z", "source": "manual"} for index, fact_id in enumerate(ids)]
    assert storage.save(data, expected_revision=data["revision"], **SCOPE)
    assert all(fact["createdAt"] == "2000-01-01T00:00:00Z" for fact in storage.load(**SCOPE)["facts"])
    target = ids[0]
    if operation == "stale_remove":
        payload = {"staleFactsToRemove": [{"id": target, "reason": "Obsolete reporting format."}]}
    elif operation == "stale_extend":
        payload = {"staleFactsToExtend": [{"id": target, "extend_by_days": 30}]}
    elif operation == "consolidate":
        payload = {
            "factsToConsolidate": [
                {
                    "sourceIds": ids[:2],
                    "consolidated": {"content": "The user prefers concise technical reports.", "confidence": 0.9, "category": "context", "scope": "user", "durability": "durable", "authority": "descriptive"},
                }
            ]
        }
    else:
        payload = {"factsToReinforce": [{"id": target, "scope": "user", "reason": "The user confirmed this preference."}]}
    model = Model(payload, lambda: peer.update_memory_fact(target, content="The user prefers a newly corrected format.", **SCOPE))
    _, updater = runtime(tmp_path, model, **options)
    messages = [HumanMessage(content="Perfect, that's exactly what I wanted!", id=MESSAGE.id)] if operation == "reinforce" else [MESSAGE]
    assert extract(updater, messages=messages) is False
    assert fresh(tmp_path)["facts"] == storage.load(**SCOPE)["facts"]
    assert next(fact for fact in fresh(tmp_path)["facts"] if fact["id"] == target)["content"] == "The user prefers a newly corrected format."
    assert storage.get_extraction_watermark(THREAD, **SCOPE) is None


def test_conflicting_batch_keeps_new_facts_and_audits_unpublished(tmp_path, monkeypatch):
    write_summary(tmp_path, text=OLD)
    payload = summary_payload()
    payload["newFacts"] = [{"content": "The user prefers primary source links.", "confidence": 0.9, "scope": "user", "durability": "durable", "authority": "descriptive"}]
    model = Model(payload, lambda: write_summary(tmp_path))
    storage, updater = runtime(tmp_path, model)
    audits = []
    monkeypatch.setattr(updater, "_record_capacity_decision", lambda *args, **kwargs: audits.append(args))
    assert extract(updater) is False
    assert fresh(tmp_path)["facts"] == []
    assert audits == []
    assert storage.get_extraction_watermark(THREAD, **SCOPE) is None
