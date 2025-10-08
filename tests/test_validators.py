from src.data.validators import FinanceMathRecord, validate_record


def sample_record() -> dict:
    return {
        "question_id": "q1",
        "question": "What is the value?",
        "topic": "finance",
        "tables_markdown": ["| A | B |\n| - | - |\n| 1 | 2 |"],
        "tables": [[{"A": "1", "B": "2"}]],
        "python_solution": "answer = 123.0",
        "ground_truth": 123.0,
    }


def test_validate_record_success():
    record = sample_record()
    validated = validate_record(record)
    assert isinstance(validated, FinanceMathRecord)
    assert validated.question_id == "q1"


def test_validate_record_missing_question():
    record = sample_record()
    record.pop("question")
    try:
        validate_record(record)
    except Exception as exc:  # ValidationError
        assert "question" in str(exc)
    else:
        raise AssertionError("Validation should fail when question is missing")
