"""Request and source-validated selection contracts for an investigation."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from scripts.inspect_sample import read_discharge_records


@dataclass(frozen=True)
class InvestigationRequest:
    """Keep requested selections explicit; None means clarification is needed."""

    question: str
    battery_id: str | None = None
    early: int | None = None
    late: int | None = None

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if self.battery_id is not None and (
            not isinstance(self.battery_id, str)
            or not self.battery_id.strip()
            or self.battery_id != self.battery_id.strip()
        ):
            raise ValueError("battery_id must be non-empty without surrounding whitespace")
        for name, value in (("early", self.early), ("late", self.late)):
            # bool is an int subclass, but True must not select discharge 1.
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.early is not None and self.late is not None and self.early >= self.late:
            raise ValueError("Choose discharge indices with early < late")

    def missing_fields(self) -> tuple[str, ...]:
        """Identify selections to clarify before looking up source records."""
        fields = (("battery_id", self.battery_id), ("early", self.early), ("late", self.late))
        return tuple(name for name, value in fields if value is None)


@dataclass(frozen=True)
class ValidatedSelection:
    """Selection identity produced by resolve_selection for one source revision."""

    battery_id: str
    source_sha256: str
    early: int
    late: int
    early_operation_index: int
    late_operation_index: int
    revision: int = 1


def resolve_selection(
    request: InvestigationRequest,
    source_path: Path,
    *,
    revision: int = 1,
) -> ValidatedSelection:
    """Validate a request against a source path supplied by trusted Python code.

    A later source registry will map a model-visible battery ID to this path.
    No report, comparison, or workflow completion marker is written here.
    """
    if request.battery_id is None or request.early is None or request.late is None:
        missing = request.missing_fields()
        raise ValueError(f"Request needs clarification: {', '.join(missing)}")
    if type(revision) is not int or revision < 1:
        raise ValueError("revision must be a positive integer")

    source_path = Path(source_path)
    if source_path.stem != request.battery_id:
        raise ValueError(
            f"Requested battery {request.battery_id} does not match source {source_path.stem}"
        )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _, records = read_discharge_records(source_path)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("Source changed while resolving the selection; retry with a stable file")
    if not 1 <= request.early < request.late <= len(records):
        raise ValueError(
            f"Choose discharge indices with 1 <= early < late <= {len(records)}; "
            f"received early={request.early}, late={request.late}"
        )

    return ValidatedSelection(
        battery_id=request.battery_id,
        source_sha256=source_sha256,
        early=request.early,
        late=request.late,
        early_operation_index=records[request.early - 1]["operation_index"],
        late_operation_index=records[request.late - 1]["operation_index"],
        revision=revision,
    )
