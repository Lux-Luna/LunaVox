"""Declarative slider group.

Legacy GUI/engine.py had a ``_setup_advanced_fields`` helper that
hand-built six near-identical (label, slider, entry) rows. :class:`ParamSliderGroup`
takes a list of :class:`FieldSpec` records and generates the same
layout from data, so adding a slider is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import customtkinter as ctk  # pyright: ignore[reportMissingImports]
except ImportError as err:  # pragma: no cover — gated by [gui] extra
    raise ImportError('customtkinter is required: pip install "lunavox[gui]"') from err

from ..theme import FONT_BODY, SPACE_MD, SPACE_SM


@dataclass(frozen=True)
class FieldSpec:
    """Description of one tunable parameter.

    ``key`` becomes the dict key in :meth:`ParamSliderGroup.values`.
    ``min_val`` / ``max_val`` / ``step`` bound the slider. ``cast`` is
    the type the raw slider value should be coerced to before being
    handed back to callers (usually ``float`` or ``int``).
    """

    key: str
    label: str
    min_val: float
    max_val: float
    step: float
    default: float
    cast: type = float


class ParamSliderGroup(ctk.CTkFrame):  # pyright: ignore[reportUntypedBaseClass]
    """A vertical stack of labelled sliders driven by :class:`FieldSpec`."""

    def __init__(self, master: Any, fields: list[FieldSpec]) -> None:
        super().__init__(master, fg_color="transparent")
        self._fields = fields
        self._vars: dict[str, Any] = {}
        self._build()

    def _build(self) -> None:
        for row, spec in enumerate(self._fields):
            ctk.CTkLabel(self, text=spec.label, font=FONT_BODY).grid(
                row=row, column=0, sticky="w", padx=(0, SPACE_MD), pady=SPACE_SM
            )
            var = ctk.DoubleVar(value=spec.default)
            self._vars[spec.key] = var

            n_steps = max(1, int(round((spec.max_val - spec.min_val) / spec.step)))
            slider = ctk.CTkSlider(
                self,
                from_=spec.min_val,
                to=spec.max_val,
                number_of_steps=n_steps,
                variable=var,
            )
            slider.grid(row=row, column=1, sticky="ew", padx=SPACE_SM, pady=SPACE_SM)

            value_label = ctk.CTkLabel(self, text=self._fmt(spec, spec.default), width=60)
            value_label.grid(row=row, column=2, sticky="e", pady=SPACE_SM)

            # Keep the readout in sync with the slider — no callbacks,
            # no double event handling: the variable itself is the bus.
            var.trace_add(
                "write",
                lambda *_a, s=spec, v=var, lbl=value_label: lbl.configure(
                    text=self._fmt(s, v.get())
                ),
            )
        self.grid_columnconfigure(1, weight=1)

    @staticmethod
    def _fmt(spec: FieldSpec, value: float) -> str:
        if spec.cast is int:
            return str(int(round(value)))
        # Use 2 decimals for small ranges and 0 for wide ones — picks
        # the precision that reads naturally for sampler vs token-count.
        return f"{value:.2f}" if spec.max_val - spec.min_val < 10 else f"{value:.0f}"

    def values(self) -> dict[str, Any]:
        """Snapshot the current slider state, cast per :class:`FieldSpec`."""
        out: dict[str, Any] = {}
        for spec in self._fields:
            raw = self._vars[spec.key].get()
            out[spec.key] = spec.cast(raw) if spec.cast is int else spec.cast(raw)
        return out

    def set_values(self, values: dict[str, float]) -> None:
        for key, val in values.items():
            if key in self._vars:
                self._vars[key].set(float(val))

    # Introspection used by tests — exposes the specs without forcing
    # callers to keep their own copy.
    @property
    def field_specs(self) -> list[FieldSpec]:
        return list(self._fields)

    def get_var(self, key: str) -> Optional[Any]:
        return self._vars.get(key)
