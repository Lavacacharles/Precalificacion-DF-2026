"""Convert a CSV file to an Excel workbook (.xlsx)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_excel(
    csv_path: Path,
    excel_path: Path,
    sheet_name: str = "Data",
    encoding: str = "utf-8-sig",
) -> None:
    """Read a CSV file and save its contents in an Excel worksheet."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if excel_path.suffix.lower() != ".xlsx":
        raise ValueError("The output file must use the .xlsx extension.")

    # Let pandas detect common delimiters such as commas, semicolons, and tabs.
    dataframe = pd.read_csv(csv_path, sep=None, engine="python", encoding=encoding)

    # Excel worksheets support at most 1,048,576 rows, including the header row.
    if len(dataframe) + 1 > 1_048_576:
        raise ValueError("The CSV has more rows than a single Excel sheet supports.")

    # Create parent directories when the requested output folder does not exist.
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(excel_path, sheet_name=sheet_name, index=False)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to an Excel .xlsx file."
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "excel_file",
        type=Path,
        nargs="?",
        help="Path to the output .xlsx file (default: input filename with .xlsx).",
    )
    parser.add_argument(
        "--sheet-name",
        default="Data",
        help='Excel worksheet name (default: "Data").',
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help='CSV text encoding (default: "utf-8-sig").',
    )
    return parser.parse_args()


def main() -> None:
    """Run the CSV-to-Excel conversion from the command line."""
    args = parse_arguments()
    output_path = args.excel_file or args.csv_file.with_suffix(".xlsx")

    try:
        convert_csv_to_excel(
            csv_path=args.csv_file,
            excel_path=output_path,
            sheet_name=args.sheet_name,
            encoding=args.encoding,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Excel file created: {output_path.resolve()}")


if __name__ == "__main__":
    main()
