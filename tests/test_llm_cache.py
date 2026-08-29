from agents.llm_cache import cache_key, get_cached, set_cached


def test_get_cached_miss_returns_none(tmp_path):
    assert get_cached("model-a", ["system", "human"], cache_dir=tmp_path) is None


def test_set_then_get_round_trips(tmp_path):
    set_cached("model-a", ["system", "human"], {"decision": {"x": 1}}, cache_dir=tmp_path)
    result = get_cached("model-a", ["system", "human"], cache_dir=tmp_path)
    assert result == {"decision": {"x": 1}}


def test_cache_key_differs_by_model():
    assert cache_key("model-a", ["same", "prompt"]) != cache_key("model-b", ["same", "prompt"])


def test_cache_key_differs_by_prompt_content():
    assert cache_key("model-a", ["prompt-1"]) != cache_key("model-a", ["prompt-2"])


def test_cache_key_stable_for_identical_input():
    assert cache_key("model-a", ["system", "human"]) == cache_key("model-a", ["system", "human"])


def test_different_prompts_do_not_collide_in_storage(tmp_path):
    set_cached("model-a", ["prompt-1"], {"decision": "first"}, cache_dir=tmp_path)
    set_cached("model-a", ["prompt-2"], {"decision": "second"}, cache_dir=tmp_path)
    assert get_cached("model-a", ["prompt-1"], cache_dir=tmp_path) == {"decision": "first"}
    assert get_cached("model-a", ["prompt-2"], cache_dir=tmp_path) == {"decision": "second"}
