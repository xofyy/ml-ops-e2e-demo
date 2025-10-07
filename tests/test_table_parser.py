from src.utils.table_parser import markdown_table_to_dataframe, table_to_records


def test_markdown_table_to_dataframe():
    markdown = """
    | Year | Revenue |
    | ---- | ------- |
    | 2022 | 1000 |
    | 2023 | 1200 |
    """
    df = markdown_table_to_dataframe(markdown)
    assert list(df.columns) == ["Year", "Revenue"]
    assert df.shape == (2, 2)
    assert df.iloc[0]["Revenue"] == "1000"


def test_table_to_records_handles_empty():
    assert table_to_records("") == []
