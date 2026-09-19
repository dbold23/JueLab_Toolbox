"""
Ground-truth / pipeline-results matching.

One shared implementation of the join that every consumer of the synthetic
benchmark used to reimplement for itself (``pipeline_validator.py`` did it by
regex + inner join, ``comparison_report.py`` did it again with an index
fallback, ``validation_summary.py`` a third time with a lower-cased string
decode of ``is_good``).

Finding M19
-----------
The old join was a silent ``how='inner'`` merge whose only signal was a
``print``.  Two failure modes were reproduced against the committed benchmark:

* renaming ``CURVE`` to ``WELL`` in the results made the merge return zero
  rows; the validator then wrote ``accuracy: 0.0`` into ``validation_report``
  and exited 0, so step 9 reported success while publishing a report claiming
  the pipeline is 0 % accurate;
* truncating the results to their first 100 rows produced ``accuracy 0.900``
  over 21 % of the benchmark, again with exit code 0 and no warning.

``match_results`` turns both into a :class:`BenchmarkMatchError`.  The
completeness counters are also attached to the merged frame's ``.attrs`` so
that a report written with ``strict=False`` still carries the evidence that it
is partial, rather than looking like a full run.

Join key
--------
The key is the ``CURVE\\d+`` token, the same one used by
``pipeline_validator.py:177`` and ``scripts/09_train_classifier.py:356``; it is
the only identifier that survives the pipeline's strain naming
(``SYNTHETIC_400pts_99h_CURVE0001`` -> ``CURVE0001``).  It is deliberately NOT
changed here: the trainer joins on the same token.

Suffixes
--------
The merge keeps the historical ``('_gt', '_pred')`` suffixes.  Only
``delta_od``, ``initial_od`` and ``max_od`` actually collide, and every
downstream reference (``expected_class``, ``true_A``, ``is_good``,
``gompertz_a`` ...) reads an unsuffixed name, so a blanket ``gt_``/``pred_``
rename would break the callers this module exists to unify.  The two decoded
verdict columns are added under the explicit names ``gt_is_good`` and
``pred_is_good``.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CURVE_ID_PATTERN",
    "CURVE_ID_RE",
    "BenchmarkMatchError",
    "extract_curve_id",
    "as_bool",
    "match_results",
]


#: The one join key.  Shared with ``09_train_classifier.py:356`` and
#: ``pipeline_validator.py:177``; do not change it without changing those.
CURVE_ID_PATTERN = r"(CURVE\d+)"
CURVE_ID_RE = re.compile(CURVE_ID_PATTERN, re.IGNORECASE)

#: Candidate identifier columns, in the order they are tried.
_GT_ID_COLUMNS = ("strain_name", "strain", "curve_id")
_RESULT_ID_COLUMNS = ("strain", "strain_name", "curve_id")

_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0", "good"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0", "bad"}

_MAX_NAMES_IN_MESSAGE = 20


class BenchmarkMatchError(RuntimeError):
    """Raised when ground truth and pipeline results do not line up 1:1.

    The offending identifiers are kept on the exception so a caller can report
    them without re-parsing the message.
    """

    def __init__(
        self,
        message: str,
        *,
        unmatched_ground_truth: Optional[Sequence[str]] = None,
        unmatched_results: Optional[Sequence[str]] = None,
        duplicate_ids: Optional[Sequence[str]] = None,
        undecided_ids: Optional[Sequence[str]] = None,
    ):
        super().__init__(message)
        self.unmatched_ground_truth: List[str] = list(unmatched_ground_truth or [])
        self.unmatched_results: List[str] = list(unmatched_results or [])
        self.duplicate_ids: List[str] = list(duplicate_ids or [])
        self.undecided_ids: List[str] = list(undecided_ids or [])


def extract_curve_id(name) -> str:
    """Return the ``CURVE\\d+`` token of ``name``, upper-cased.

    Falls back to the stripped, upper-cased name when the token is absent, so
    that a non-synthetic identifier still joins to itself; the completeness
    check in :func:`match_results` is what catches a genuinely wrong naming
    scheme.
    """
    text = "" if name is None else str(name)
    text = text.strip().upper()
    hit = CURVE_ID_RE.search(text)
    return hit.group(1) if hit else text


def _id_series(frame: pd.DataFrame, candidates: Iterable[str], what: str) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column].map(extract_curve_id)
    raise BenchmarkMatchError(
        f"{what} has none of the identifier columns {tuple(candidates)}; "
        f"columns present: {list(frame.columns)[:20]}"
    )


def as_bool(values) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a verdict column to ``(bool array, undecided mask)``.

    Accepts ``bool``, numeric and the string spellings the pipeline and its CSV
    round-trip produce.  Anything unrecognised -- ``NaN``, an empty cell, a
    stray word -- decodes to ``False`` and is flagged in the undecided mask
    instead of raising here, so the caller decides whether a blank verdict is
    fatal (``strict=True``) or scored as "not good" (``strict=False``).
    """
    series = pd.Series(values).reset_index(drop=True)
    out = np.zeros(len(series), dtype=bool)
    undecided = np.zeros(len(series), dtype=bool)

    for i, raw in enumerate(series):
        if isinstance(raw, (bool, np.bool_)):
            out[i] = bool(raw)
            continue
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            undecided[i] = True
            continue
        if isinstance(raw, (int, float, np.integer, np.floating)):
            if isinstance(raw, float) and np.isnan(raw):
                undecided[i] = True
            else:
                out[i] = bool(raw)
            continue
        token = str(raw).strip().lower()
        if token in _TRUE_TOKENS:
            out[i] = True
        elif token in _FALSE_TOKENS or token in ("", "nan", "none", "<na>"):
            out[i] = False
            if token in ("", "nan", "none", "<na>"):
                undecided[i] = True
        else:
            undecided[i] = True
    return out, undecided


def _names(items: Sequence[str]) -> str:
    items = list(items)
    shown = ", ".join(items[:_MAX_NAMES_IN_MESSAGE])
    if len(items) > _MAX_NAMES_IN_MESSAGE:
        shown += f", ... ({len(items)} total)"
    return shown


def match_results(
    ground_truth_df: pd.DataFrame,
    results_df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """Join pipeline results to ground truth on the ``CURVE\\d+`` token.

    Parameters
    ----------
    ground_truth_df, results_df
        The two frames.  Either may carry the v1 or the v2 ground-truth
        columns; only the identifier column and ``expected_class`` /
        ``is_good`` are required.
    strict
        When true (the default) any unmatched ground-truth id, unmatched
        results id, duplicated id or undecided verdict raises
        :class:`BenchmarkMatchError`.  ``strict=False`` is the explicit escape
        hatch for a deliberately partial run: the same counters are still
        attached to ``merged.attrs`` so the resulting report is self-labelling.

    Returns
    -------
    pandas.DataFrame
        The inner join, with ``('_gt', '_pred')`` suffixes preserved, plus:

        ``curve_id``
            the normalised join key (the frame's original ``curve_id``, if it
            had one, is kept as ``ground_truth_curve_id``);
        ``gt_is_good``
            ``expected_class == 'GOOD'``;
        ``pred_is_good``
            the decoded pipeline verdict.

        ``merged.attrs`` carries ``n_ground_truth``, ``n_results``,
        ``n_matched``, ``n_unmatched_ground_truth``, ``n_unmatched_results``,
        ``n_duplicate_ids``, ``n_undecided``, ``coverage`` and the
        corresponding id lists (``unmatched_ground_truth_ids``,
        ``unmatched_results_ids``, ``duplicate_ids``, ``undecided_ids``).
    """
    gt = ground_truth_df.copy()
    pr = results_df.copy()

    gt_key = _id_series(gt, _GT_ID_COLUMNS, "ground truth")
    pr_key = _id_series(pr, _RESULT_ID_COLUMNS, "pipeline results")

    gt["_match_key"] = gt_key.values
    pr["_match_key"] = pr_key.values

    gt_dupes = sorted(gt["_match_key"][gt["_match_key"].duplicated()].unique().tolist())
    pr_dupes = sorted(pr["_match_key"][pr["_match_key"].duplicated()].unique().tolist())
    duplicate_ids = sorted(set(gt_dupes) | set(pr_dupes))

    gt_ids = set(gt["_match_key"])
    pr_ids = set(pr["_match_key"])
    unmatched_gt = sorted(gt_ids - pr_ids)
    unmatched_pr = sorted(pr_ids - gt_ids)

    merged = gt.merge(pr, on="_match_key", how="inner", suffixes=("_gt", "_pred"))

    # Normalised identifier.  The ground truth's own ``curve_id`` is an integer
    # row index, not the join key, so it is preserved under a distinct name
    # rather than silently overwritten.
    if "curve_id" in merged.columns:
        merged = merged.rename(columns={"curve_id": "ground_truth_curve_id"})
    merged.insert(0, "curve_id", merged["_match_key"].values)
    merged = merged.drop(columns=["_match_key"])

    if "expected_class" in merged.columns:
        merged["gt_is_good"] = (
            merged["expected_class"].astype("string").str.strip().str.upper() == "GOOD"
        ).fillna(False).to_numpy(dtype=bool)
    else:
        raise BenchmarkMatchError(
            "ground truth has no 'expected_class' column; columns present: "
            f"{list(ground_truth_df.columns)[:20]}"
        )

    if "is_good" in merged.columns:
        decoded, undecided = as_bool(merged["is_good"])
    else:
        raise BenchmarkMatchError(
            "pipeline results have no 'is_good' column; columns present: "
            f"{list(results_df.columns)[:20]}"
        )
    merged["pred_is_good"] = decoded
    undecided_ids = merged.loc[undecided, "curve_id"].tolist()

    n_gt = len(gt)
    merged.attrs["n_ground_truth"] = n_gt
    merged.attrs["n_results"] = len(pr)
    merged.attrs["n_matched"] = len(merged)
    merged.attrs["n_unmatched_ground_truth"] = len(unmatched_gt)
    merged.attrs["n_unmatched_results"] = len(unmatched_pr)
    merged.attrs["n_duplicate_ids"] = len(duplicate_ids)
    merged.attrs["n_undecided"] = len(undecided_ids)
    merged.attrs["coverage"] = (len(merged) / n_gt) if n_gt else 0.0
    merged.attrs["unmatched_ground_truth_ids"] = unmatched_gt
    merged.attrs["unmatched_results_ids"] = unmatched_pr
    merged.attrs["duplicate_ids"] = duplicate_ids
    merged.attrs["undecided_ids"] = undecided_ids
    merged.attrs["strict"] = bool(strict)

    if strict:
        problems = []
        if unmatched_gt:
            problems.append(
                f"{len(unmatched_gt)} ground-truth ids have no pipeline result: "
                f"{_names(unmatched_gt)}"
            )
        if unmatched_pr:
            problems.append(
                f"{len(unmatched_pr)} pipeline results have no ground truth: "
                f"{_names(unmatched_pr)}"
            )
        if duplicate_ids:
            problems.append(
                f"{len(duplicate_ids)} duplicated curve ids: {_names(duplicate_ids)}"
            )
        if undecided_ids:
            problems.append(
                f"{len(undecided_ids)} curves have an undecodable is_good verdict: "
                f"{_names(undecided_ids)}"
            )
        if problems:
            raise BenchmarkMatchError(
                "ground truth and pipeline results do not match 1:1 "
                f"(matched {len(merged)} of {n_gt} ground-truth curves, "
                f"coverage {merged.attrs['coverage']:.3f}). "
                + " | ".join(problems),
                unmatched_ground_truth=unmatched_gt,
                unmatched_results=unmatched_pr,
                duplicate_ids=duplicate_ids,
                undecided_ids=undecided_ids,
            )

    return merged
