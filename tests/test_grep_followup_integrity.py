"""Search errors and follow-ups over real disposable filesystem fixtures."""
import asyncio
import json
import os
import shutil

import pytest

from src.agent_tools.filesystem_tools import GrepTool


def search(workspace, pattern, **args):
    from src.tool_execution import _active_workspace
    async def run():
        token = _active_workspace.set(str(workspace))
        try:
            return await GrepTool().execute(json.dumps({'pattern': pattern, **args}), {})
        finally:
            _active_workspace.reset(token)
    return asyncio.run(run())


def test_invalid_regex_is_error_not_empty_search(tmp_path):
    if not shutil.which('rg'):
        pytest.skip('Requires ripgrep backend')
    (tmp_path / 'sample.txt').write_text('alpha\nbeta\n', encoding='utf8')
    result = search(tmp_path, '[')
    assert result['exit_code'] == 1
    assert 'error' in result
    assert 'No matches' not in str(result)


def test_fallback_unreadable_file_is_error_not_empty(tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip('Root bypasses filesystem read permission fixture')
    path = tmp_path / 'unreadable.txt'
    path.write_text('MATCH\n', encoding='utf8')
    path.chmod(0)
    real_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda name, *args, **kwargs: None if name == 'rg' else real_which(name, *args, **kwargs))
    try:
        result = search(tmp_path, 'MATCH')
        assert result['exit_code'] == 1
        assert 'error' in result
    finally:
        path.chmod(0o600)
    assert 'No matches' not in str(result)


def test_single_file_search_retains_filename_and_line_for_followup(tmp_path):
    (tmp_path / 'sample.txt').write_text('alpha\nbeta\n', encoding='utf8')
    result = search(tmp_path, 'beta', path='/workspace/sample.txt')
    assert result['exit_code'] == 0
    assert '/workspace/sample.txt:2:beta' in result['output']
    assert str(tmp_path) not in result['output']


def test_fallback_does_not_follow_file_symlink_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'inside.txt').write_text('MATCH_INSIDE\n', encoding='utf8')
    outside = tmp_path / 'outside.txt'
    outside.write_text('MATCH_OUTSIDE\n', encoding='utf8')
    (workspace / 'escape.txt').symlink_to(outside)
    real_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda name, *args, **kwargs: None if name == 'rg' else real_which(name, *args, **kwargs))
    result = search(workspace, 'MATCH')
    assert result['exit_code'] == 0
    assert 'MATCH_INSIDE' in result['output']
    assert 'MATCH_OUTSIDE' not in result['output']


@pytest.mark.parametrize('fallback', [False, True])
def test_missing_search_target_is_not_reported_empty(tmp_path, monkeypatch, fallback):
    if fallback:
        real_which = shutil.which
        monkeypatch.setattr(shutil, 'which', lambda name, *args, **kwargs: None if name == 'rg' else real_which(name, *args, **kwargs))
    result = search(tmp_path, 'alpha', path='/workspace/missing.txt')
    assert result['exit_code'] == 1
    assert 'error' in result


@pytest.mark.parametrize('fallback', [False, True])
def test_error_then_case_refinement_and_genuine_empty_result(tmp_path, monkeypatch, fallback):
    if fallback:
        real_which = shutil.which
        monkeypatch.setattr(shutil, 'which', lambda name, *args, **kwargs: None if name == 'rg' else real_which(name, *args, **kwargs))
    path = tmp_path / 'sample.txt'
    original = b'Alpha\nalpha\nviolet-72\n'
    path.write_bytes(original)
    assert search(tmp_path, '[')['exit_code'] == 1
    exact = search(tmp_path, 'alpha', path='/workspace/sample.txt')
    assert exact['exit_code'] == 0
    assert '/workspace/sample.txt:2:alpha' in exact['output']
    assert ':1:Alpha' not in exact['output']
    refined = search(tmp_path, 'alpha', path='/workspace/sample.txt', ignore_case=True)
    assert refined['exit_code'] == 0 and ':1:Alpha' in refined['output'] and ':2:alpha' in refined['output']
    empty = search(tmp_path, 'no-such-marker')
    assert empty['exit_code'] == 0 and 'No matches' in empty['output']
    assert path.read_bytes() == original
