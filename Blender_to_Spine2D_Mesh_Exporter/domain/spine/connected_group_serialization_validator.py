"""Serialization validator for already validated connected Spine documents.

Historical connected JSON from ``main`` intentionally assigns one constraint order to a
whole Z layer and leaves every legacy scale compensator at its standalone order. The
normal serializer must remain strict for every other document, so this specialization
filters only the resulting duplicate-order diagnostic and delegates all other structural,
mesh, weighted-stream, animation, and cross-reference validation unchanged.
"""

from __future__ import annotations

from .model import SpineDocument
from .serialization_validator import SpineSerializationValidator
from .validator import SpineValidationIssue


class ConnectedGroupSerializationValidator(SpineSerializationValidator):
    """Permit only Legacy-compatible connected order ties during serialization."""

    def validate(self, document: SpineDocument) -> tuple[SpineValidationIssue, ...]:
        issues = super().validate(document)
        return tuple(
            issue
            for issue in issues
            if issue.code != "DUPLICATE_CONSTRAINT_ORDER"
        )


__all__ = ["ConnectedGroupSerializationValidator"]
