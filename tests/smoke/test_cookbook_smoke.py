"""Cookbook: hardware is detected and the recommendations are sized against it.

What the README advertises here is hardware-aware recommendation, and
that is exactly the part that runs offline. Downloading and serving a
model is left to the gap list: it needs tmux, a GPU runtime and several
gigabytes over the network.
"""
from __future__ import annotations

SYSTEM_PATH = "/api/hwfit/system"
MODELS_PATH = "/api/hwfit/models"
STATE_PATH = "/api/cookbook/state"
GPUS_PATH = "/api/cookbook/gpus"

STATE_MARKER = "odysseusSmokeMarker"


def test_hardware_is_detected(client):
    response = client.get(SYSTEM_PATH)
    assert response.status_code == 200, response.text
    system = response.json()
    assert (system.get("total_ram_gb") or 0) > 0, system
    assert (system.get("cpu_cores") or 0) > 0, system
    assert system.get("cpu_name"), system

    gpus = client.get(GPUS_PATH)
    assert gpus.status_code == 200, gpus.text
    assert gpus.json().get("ok") is True, gpus.text


def test_recommendations_fit_the_detected_hardware(client):
    response = client.get(MODELS_PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    system = body.get("system") or {}
    assert system.get("cpu_name"), body
    recommended = body.get("models") or body.get("recommendations") or []
    assert recommended, f"no model recommendation for this hardware: {list(body)}"


def test_cookbook_state_persists(client):
    written = client.post(STATE_PATH, json={STATE_MARKER: "ody-95"})
    assert written.status_code == 200, written.text
    assert written.json().get("ok") is True, written.text

    read = client.get(STATE_PATH)
    assert read.status_code == 200, read.text
    assert read.json().get(STATE_MARKER) == "ody-95", read.text

    client.post(STATE_PATH, json={})
