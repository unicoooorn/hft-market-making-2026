#!/usr/bin/env python3
"""Convert CSV market data files to Parquet format with scaled integers."""

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as csv
import pyarrow.compute as pc

SCALE = 10**8


def convert_lob_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    row_group_size: int = 100000,
) -> None:
    """Convert order book CSV to Parquet with scaled int64 prices/amounts."""
    print(f"Converting {csv_path} to {parquet_path}...")

    table = csv.read_csv(
        csv_path,
        convert_options=csv.ConvertOptions(
            column_types={
                "": pa.int64(),
                "local_timestamp": pa.int64(),
                "bids[0].price": pa.float64(),
                "bids[0].amount": pa.float64(),
                "asks[0].price": pa.float64(),
                "asks[0].amount": pa.float64(),
            }
        ),
    )

    # Scale price/amount columns to int64
    schema = table.schema
    new_columns = []
    for i, field in enumerate(schema):
        col = table.column(i)
        if field.name in ("bids[0].price", "bids[0].amount", "asks[0].price", "asks[0].amount"):
            scaled = pc.multiply(pc.cast(col, pa.float64()), SCALE)
            scaled_int = pc.cast(scaled, pa.int64(), safe=False)
            new_columns.append(scaled_int)
        else:
            new_columns.append(col)

    scaled_table = pa.table(dict(zip(schema.names, new_columns)))

    pq.write_table(
        scaled_table,
        parquet_path,
        compression="snappy",
        row_group_size=row_group_size,
    )

    print(f"  Rows: {table.num_rows:,}")
    print(f"  Columns: {table.num_columns}")
    print("  Done!")


def convert_trades_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    row_group_size: int = 100000,
) -> None:
    """Convert trades CSV to Parquet with scaled int64 prices/amounts."""
    print(f"Converting {csv_path} to {parquet_path}...")

    table = csv.read_csv(
        csv_path,
        convert_options=csv.ConvertOptions(
            column_types={
                "": pa.int64(),
                "local_timestamp": pa.int64(),
                "price": pa.float64(),
                "amount": pa.float64(),
            }
        ),
    )

    # Scale price/amount columns to int64
    schema = table.schema
    new_columns = []
    for i, field in enumerate(schema):
        col = table.column(i)
        if field.name in ("price", "amount"):
            scaled = pc.multiply(pc.cast(col, pa.float64()), SCALE)
            scaled_int = pc.cast(scaled, pa.int64(), safe=False)
            new_columns.append(scaled_int)
        else:
            new_columns.append(col)

    scaled_table = pa.table(dict(zip(schema.names, new_columns)))

    pq.write_table(
        scaled_table,
        parquet_path,
        compression="snappy",
        row_group_size=row_group_size,
    )

    print(f"  Rows: {table.num_rows:,}")
    print(f"  Columns: {table.num_columns}")
    print("  Done!")


def main() -> None:
    """Convert all CSV files to Parquet."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "cmf"

    convert_lob_csv_to_parquet(
        data_dir / "lob.csv",
        data_dir / "lob.parquet",
    )

    convert_trades_csv_to_parquet(
        data_dir / "trades.csv",
        data_dir / "trades.parquet",
    )

    print("\nAll files converted successfully!")


if __name__ == "__main__":
    main()
