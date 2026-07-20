import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.blender_registration import (
    RegistrationCleanupAction,
    RegistrationCleanupError,
    RnaPropertyRegistration,
    register_classes_transactionally,
    register_rna_properties_transactionally,
    unregister_all_best_effort,
)


class _ClassA:
    pass


class _ClassB:
    pass


class _ClassC:
    pass


def test_class_failure_rolls_back_registered_prefix_in_reverse_order():
    events = []

    def register(cls):
        events.append(("register", cls.__name__))
        if cls is _ClassC:
            raise RuntimeError("boom")

    def unregister(cls):
        events.append(("unregister", cls.__name__))

    with pytest.raises(RuntimeError, match="boom"):
        register_classes_transactionally(
            (_ClassA, _ClassB, _ClassC),
            register_class=register,
            unregister_class=unregister,
        )

    assert events == [
        ("register", "_ClassA"),
        ("register", "_ClassB"),
        ("register", "_ClassC"),
        ("unregister", "_ClassB"),
        ("unregister", "_ClassA"),
    ]


def test_cleanup_failure_is_structured_and_keeps_primary_error():
    def register(cls):
        if cls is _ClassB:
            raise ValueError("register failed")

    def unregister(_cls):
        raise RuntimeError("cleanup failed")

    with pytest.raises(RegistrationCleanupError) as captured:
        register_classes_transactionally(
            (_ClassA, _ClassB),
            register_class=register,
            unregister_class=unregister,
        )

    error = captured.value
    assert isinstance(error.primary_error, ValueError)
    assert len(error.failures) == 1
    assert error.failures[0].label.endswith("._ClassA")


def test_rna_failure_removes_only_properties_added_by_this_call():
    class Owner:
        pass

    class FailingOwner:
        def __setattr__(self, name, value):
            if name == "bad":
                raise RuntimeError("assignment failed")
            super().__setattr__(name, value)

    good_owner = Owner()
    bad_owner = FailingOwner()
    registrations = (
        RnaPropertyRegistration(good_owner, "good", object()),
        RnaPropertyRegistration(bad_owner, "bad", object()),
    )

    with pytest.raises(RuntimeError, match="assignment failed"):
        register_rna_properties_transactionally(registrations)

    assert not hasattr(good_owner, "good")
    assert not hasattr(bad_owner, "bad")


def test_existing_rna_property_is_preserved_after_rollback():
    class Owner:
        existing = "foreign"

    owner = Owner()
    registrations = (
        RnaPropertyRegistration(owner, "new_property", 1),
        RnaPropertyRegistration(owner, "existing", 2),
    )

    with pytest.raises(ValueError, match="already present"):
        register_rna_properties_transactionally(registrations)

    assert not hasattr(owner, "new_property")
    assert owner.existing == "foreign"


def test_best_effort_cleanup_runs_every_action_before_raising():
    events = []

    def fail_first():
        events.append("first")
        raise RuntimeError("first failed")

    def succeed_second():
        events.append("second")

    actions = (
        RegistrationCleanupAction("first", fail_first),
        RegistrationCleanupAction("second", succeed_second),
    )

    with pytest.raises(RegistrationCleanupError) as captured:
        unregister_all_best_effort(actions, operation="test cleanup")

    assert events == ["first", "second"]
    assert [item.label for item in captured.value.failures] == ["first"]


def test_duplicate_class_and_rna_identity_fail_before_side_effects():
    events = []
    with pytest.raises(ValueError, match="duplicate"):
        register_classes_transactionally(
            (_ClassA, _ClassA),
            register_class=lambda cls: events.append(cls),
            unregister_class=lambda _cls: None,
        )
    assert events == []

    class Owner:
        pass

    owner = Owner()
    item = RnaPropertyRegistration(owner, "prop", 1)
    with pytest.raises(ValueError, match="duplicate"):
        register_rna_properties_transactionally((item, item))
    assert not hasattr(owner, "prop")
