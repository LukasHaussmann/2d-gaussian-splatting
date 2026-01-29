"""
LaTeX formatting utilities for generating publication-ready tables.
"""

from typing import List, Optional, Tuple, Union
import re


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


def bold(text: str) -> str:
    """Wrap text in LaTeX bold command."""
    return f"\\textbf{{{text}}}"


def underline(text: str) -> str:
    """Wrap text in LaTeX underline command."""
    return f"\\underline{{{text}}}"


def format_metric(
    value: float,
    precision: int = 2,
    is_best: bool = False,
    is_second: bool = False
) -> str:
    """
    Format a metric value for LaTeX table.

    Parameters
    ----------
    value : float
        The metric value
    precision : int
        Number of decimal places
    is_best : bool
        If True, wrap in bold
    is_second : bool
        If True, wrap in underline

    Returns
    -------
    str
        Formatted metric string
    """
    if value is None or value != value:  # Check for None and NaN
        return "-"

    formatted = f"{value:.{precision}f}"

    if is_best:
        return bold(formatted)
    elif is_second:
        return underline(formatted)
    return formatted


def format_metric_with_std(
    mean: float,
    std: float,
    precision: int = 2,
    is_best: bool = False,
    is_second: bool = False
) -> str:
    """
    Format a metric value with standard deviation for LaTeX table.

    Parameters
    ----------
    mean : float
        Mean value
    std : float
        Standard deviation
    precision : int
        Number of decimal places
    is_best : bool
        If True, wrap in bold
    is_second : bool
        If True, wrap in underline

    Returns
    -------
    str
        Formatted string like "0.45 ± 0.03"
    """
    if mean != mean:  # Check for NaN
        return "-"

    formatted = f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"

    if is_best:
        return bold(formatted)
    elif is_second:
        return underline(formatted)
    return formatted


def make_table_header(
    columns: List[str],
    caption: str = "",
    label: str = "",
    column_format: Optional[str] = None
) -> str:
    """
    Generate LaTeX table header.

    Parameters
    ----------
    columns : List[str]
        Column headers
    caption : str
        Table caption
    label : str
        Table label for referencing
    column_format : str, optional
        LaTeX column format (e.g., "l|ccc")

    Returns
    -------
    str
        LaTeX table header
    """
    if column_format is None:
        column_format = "l" + "c" * (len(columns) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]

    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    lines.extend([
        f"\\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ])

    return "\n".join(lines)


def make_table_footer() -> str:
    """Generate LaTeX table footer."""
    return "\n".join([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def make_table_row(values: List[str], hline_after: bool = False) -> str:
    """
    Generate a LaTeX table row.

    Parameters
    ----------
    values : List[str]
        Cell values
    hline_after : bool
        If True, add horizontal line after row

    Returns
    -------
    str
        LaTeX table row
    """
    row = " & ".join(values) + r" \\"
    if hline_after:
        row += "\n\\midrule"
    return row


def make_multirow(text: str, nrows: int) -> str:
    """Create a multirow cell."""
    return f"\\multirow{{{nrows}}}{{*}}{{{text}}}"


def make_multicolumn(text: str, ncols: int, alignment: str = "c") -> str:
    """Create a multicolumn cell."""
    return f"\\multicolumn{{{ncols}}}{{{alignment}}}{{{text}}}"


def find_best_and_second(
    values: List[float],
    lower_is_better: bool = True
) -> Tuple[int, int]:
    """
    Find indices of best and second-best values.

    Parameters
    ----------
    values : List[float]
        List of metric values
    lower_is_better : bool
        If True, lower values are better

    Returns
    -------
    Tuple[int, int]
        Indices of best and second-best values (-1 if not found)
    """
    # Filter out None and NaN values
    valid_indices = [i for i, v in enumerate(values) if v is not None and v == v]

    if len(valid_indices) == 0:
        return -1, -1
    if len(valid_indices) == 1:
        return valid_indices[0], -1

    if lower_is_better:
        sorted_indices = sorted(valid_indices, key=lambda i: values[i])
    else:
        sorted_indices = sorted(valid_indices, key=lambda i: -values[i])

    return sorted_indices[0], sorted_indices[1]


def format_column_with_ranking(
    values: List[float],
    precision: int = 2,
    lower_is_better: bool = True
) -> List[str]:
    """
    Format a column of values with best/second-best highlighting.

    Parameters
    ----------
    values : List[float]
        Metric values
    precision : int
        Decimal precision
    lower_is_better : bool
        If True, lower values are better

    Returns
    -------
    List[str]
        Formatted strings with bold/underline for best/second-best
    """
    best_idx, second_idx = find_best_and_second(values, lower_is_better)

    formatted = []
    for i, v in enumerate(values):
        is_best = (i == best_idx)
        is_second = (i == second_idx)
        formatted.append(format_metric(v, precision, is_best, is_second))

    return formatted


def create_comparison_table(
    methods: List[str],
    metrics: dict,
    caption: str = "",
    label: str = ""
) -> str:
    """
    Create a complete comparison table.

    Parameters
    ----------
    methods : List[str]
        Method names
    metrics : dict
        Dictionary mapping metric names to lists of (mean, std) tuples
    caption : str
        Table caption
    label : str
        Table label

    Returns
    -------
    str
        Complete LaTeX table
    """
    metric_names = list(metrics.keys())
    columns = ["Method"] + metric_names

    # Find best for each metric
    best_indices = {}
    second_indices = {}
    for metric_name in metric_names:
        values = [m[0] for m in metrics[metric_name]]  # Extract means
        best_indices[metric_name], second_indices[metric_name] = \
            find_best_and_second(values, lower_is_better=True)

    lines = [make_table_header(columns, caption, label)]

    for i, method in enumerate(methods):
        row_values = [escape_latex(method)]
        for metric_name in metric_names:
            mean, std = metrics[metric_name][i]
            is_best = (i == best_indices[metric_name])
            is_second = (i == second_indices[metric_name])
            row_values.append(format_metric_with_std(mean, std, 2, is_best, is_second))
        lines.append(make_table_row(row_values))

    lines.append(make_table_footer())

    return "\n".join(lines)
