from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class IterationResult:
    """Represents a single iteration result."""

    # Core fields
    iter: int
    time: float
    valid: bool
    config7_cost: float
    solver_cost: float
    easy_cost: float

    # Optional fields with defaults
    number_of_violations: Optional[int] = None
    total_arcs: Optional[int] = None
    classifier_score: Optional[float] = None
    entropy_score: Optional[float] = None
    top_arc_value: Optional[float] = None
    top_3_arcs: Optional[str] = None  # Assuming this is a string representation of arcs
    entropy_previous: Optional[float] = None
    classifier_score_previous: Optional[float] = None
    first_time_valid: Optional[bool] = None  # Assuming this is an optional boolean
    top_arc_previous: Optional[float] = None  # Assuming this is an optional float
    top_3_arcs_previous: Optional[str] = None

    # Computed fields
    entropy_variation: Optional[float] = None
    classifier_score_variation: Optional[float] = None
    top_arc_variation: Optional[float] = None
    top_3_arcs_variation: Optional[float] = None

    # Store additional dynamic attributes
    additional_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pandas_series(cls, row: pd.Series) -> "IterationResult":
        """Create IterationResult from pandas Series with validation."""

        # Required fields mapping
        required_fields = {
            "iter": int,
            "time": float,
            "valid": bool,
            "config7_cost": float,
            "solver_cost": float,
            "easy_cost": float,
        }

        # Optional fields mapping
        optional_fields = {
            "number_of_violations": int,
            "entropy_score": float,
            "classifier_score": float,
            "top_arc_value": float,
            "top_3_arcs": float,
            "entropy_previous": float,
            "classifier_score_previous": float,
            "first_time_valid": bool,  # Assuming this is an optional int
            "top_arc_previous": float,  # Assuming this is an optional float
            "top_3_arcs_previous": float,  # Assuming this is an optional string
        }

        computed_fields = {
            "entropy_variation": float,
            "classifier_score_variation": float,
            "top_arc_variation": float,
            "top_3_arcs_variation": float,  # Assuming this is a string representation of arcs
        }

        # Extract and validate required fields
        kwargs = {}
        for field_name, field_type in required_fields.items():
            if field_name not in row:
                raise ValueError(f"Required field '{field_name}' missing from row")

            try:
                kwargs[field_name] = field_type(row[field_name])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert '{field_name}' to {field_type.__name__}: {e}"
                )

        # Extract optional fields
        for field_name, field_type in optional_fields.items():
            if field_name in row and pd.notna(row[field_name]):
                try:
                    kwargs[field_name] = field_type(row[field_name])
                except (ValueError, TypeError):
                    print(
                        f"Warning: Cannot convert '{field_name}' to {field_type.__name__}, skipping."
                    )
                    pass  # Ignore conversion errors for optional fields

        if "classifier_score_previous" not in kwargs:
            kwargs["classifier_score_previous"] = None
        if "entropy_previous" not in kwargs:
            kwargs["entropy_previous"] = None
        if "top_arc_previous" not in kwargs:
            kwargs["top_arc_previous"] = None
        if "top_3_arcs_previous" not in kwargs:
            kwargs["top_3_arcs_previous"] = None

        try:
            kwargs["entropy_variation"] = kwargs["entropy_score"] - kwargs.get(
                "entropy_previous"
            )
        except Exception:
            kwargs["entropy_variation"] = None
        try:
            kwargs["classifier_score_variation"] = kwargs[
                "classifier_score"
            ] - kwargs.get("classifier_score_previous")
        except Exception:
            kwargs["classifier_score_variation"] = None

        try:
            if "top_arc_value" in kwargs and "top_arc_previous" in kwargs:
                kwargs["top_arc_variation"] = kwargs["top_arc_value"] - kwargs.get(
                    "top_arc_previous"
                )
            else:
                kwargs["top_arc_variation"] = None
        except Exception:
            kwargs["top_arc_variation"] = None

            # Compute classifier score variation

        try:
            if "top_3_arcs" in kwargs and "top_3_arcs_previous" in kwargs:
                kwargs["top_3_arcs_variation"] = kwargs["top_3_arcs"] - kwargs.get(
                    "top_3_arcs_previous"
                )
            else:
                kwargs["top_3_arcs_variation"] = None
        except Exception:
            kwargs["top_3_arcs_variation"] = None

        # Store any additional fields
        additional_data = {}
        for key, value in row.items():
            if key not in required_fields and key not in optional_fields:
                additional_data[key] = value

        if additional_data:
            kwargs["additional_data"] = additional_data

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationResult":
        """Create IterationResult from dictionary."""
        return cls.from_pandas_series(pd.Series(data))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "iter": self.iter,
            "time": self.time,
            "valid": self.valid,
            "config7_cost": self.config7_cost,
            "solver_cost": self.solver_cost,
            "easy_cost": self.easy_cost,
        }

        # Add optional fields if they exist and are not None
        optional_fields = [
            "number_of_violations",
            "total_arcs",
            "classifier_score",
            "entropy_score",
            "top_arc_value",
            "top_3_arcs",
            "entropy_previous",
            "classifier_score_previous",
            "top_arc_previous",
            "top_3_arcs_previous",
            "first_time_valid",
            "entropy_variation",
            "classifier_score_variation",
            "top_arc_variation",
            "top_3_arcs_variation",
        ]

        for field in optional_fields:
            value = getattr(self, field, None)
            if value is not None:
                result[field] = value

        # Add additional data
        result.update(self.additional_data)

        return result
