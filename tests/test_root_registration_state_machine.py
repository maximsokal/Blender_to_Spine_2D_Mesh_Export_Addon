from types import SimpleNamespace

import pytest

import Blender_to_Spine2D_Mesh_Exporter as extension


def _prepare_root(monkeypatch, steps):
    extension._REGISTRATION_STATE = extension.ExtensionRegistrationState.UNREGISTERED
    monkeypatch.setattr(extension, "REGISTRATION_STEPS", tuple(steps))
    monkeypatch.setattr(extension, "require_supported_blender_runtime", lambda _bpy: None)
    monkeypatch.setattr(extension.config, "_setup_default_logging", lambda: None)
    monkeypatch.setattr(extension.config, "setup_logging", lambda: None)
    monkeypatch.setattr(extension, "initialize_logging_preferences", lambda _prefs: ())
    monkeypatch.setattr(
        extension.bpy.context,
        "preferences",
        SimpleNamespace(
            addons={
                extension.__name__: SimpleNamespace(preferences=object()),
            }
        ),
        raising=False,
    )


def test_root_register_and_unregister_are_idempotent(monkeypatch):
    calls = []
    steps = (
        ("first", lambda: calls.append("register:first"), lambda: calls.append("unregister:first")),
        ("second", lambda: calls.append("register:second"), lambda: calls.append("unregister:second")),
    )
    _prepare_root(monkeypatch, steps)

    extension.register()
    extension.register()
    assert extension.get_registration_state() is extension.ExtensionRegistrationState.REGISTERED
    assert calls == ["register:first", "register:second"]

    extension.unregister()
    extension.unregister()
    assert extension.get_registration_state() is extension.ExtensionRegistrationState.UNREGISTERED
    assert calls == [
        "register:first",
        "register:second",
        "unregister:second",
        "unregister:first",
    ]


@pytest.mark.parametrize("failure_index", range(4))
def test_root_registration_failure_rolls_back_completed_steps_in_reverse_order(
    monkeypatch,
    failure_index,
):
    calls = []
    steps = []
    for index in range(4):
        def register(index=index):
            calls.append(f"register:{index}")
            if index == failure_index:
                raise RuntimeError(f"forced registration failure {index}")

        def unregister(index=index):
            calls.append(f"unregister:{index}")

        steps.append((f"step-{index}", register, unregister))

    _prepare_root(monkeypatch, steps)
    with pytest.raises(RuntimeError, match=f"forced registration failure {failure_index}"):
        extension.register()

    assert extension.get_registration_state() is extension.ExtensionRegistrationState.UNREGISTERED
    assert calls == [
        *(f"register:{index}" for index in range(failure_index + 1)),
        *(f"unregister:{index}" for index in reversed(range(failure_index))),
    ]


def test_preflight_failure_does_not_leave_registering_state(monkeypatch):
    _prepare_root(monkeypatch, ())

    def fail_runtime(_bpy):
        raise RuntimeError("unsupported runtime")

    monkeypatch.setattr(extension, "require_supported_blender_runtime", fail_runtime)
    with pytest.raises(RuntimeError, match="unsupported runtime"):
        extension.register()
    assert extension.get_registration_state() is extension.ExtensionRegistrationState.UNREGISTERED


def test_cleanup_failure_marks_degraded_and_later_unregister_can_recover(monkeypatch):
    calls = []
    fail_cleanup = True

    def register_first():
        calls.append("register:first")

    def unregister_first():
        nonlocal fail_cleanup
        calls.append("unregister:first")
        if fail_cleanup:
            raise RuntimeError("forced cleanup failure")

    def register_second():
        calls.append("register:second")
        raise RuntimeError("forced primary failure")

    def unregister_second():
        calls.append("unregister:second")

    steps = (
        ("first", register_first, unregister_first),
        ("second", register_second, unregister_second),
    )
    _prepare_root(monkeypatch, steps)

    with pytest.raises(Exception, match="forced cleanup failure"):
        extension.register()
    assert extension.get_registration_state() is extension.ExtensionRegistrationState.DEGRADED

    with pytest.raises(RuntimeError, match="degraded registration state"):
        extension.register()

    fail_cleanup = False
    extension.unregister()
    assert extension.get_registration_state() is extension.ExtensionRegistrationState.UNREGISTERED
    assert calls[-2:] == ["unregister:second", "unregister:first"]
