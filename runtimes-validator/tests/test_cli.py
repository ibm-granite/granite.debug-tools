from unittest.mock import patch

from runtimes_validator.cli import main


def test_list_engines(capsys):
    rc = main(["--list-engines"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "ollama" in output
    assert "vllm" in output
    assert "llamacpp" in output


def test_list_tests(capsys):
    rc = main(["--list-tests"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "basic_generation" in output


def test_missing_engine_errors(capsys):
    try:
        main(["--model", "some-model"])
        assert False, "Should have exited"
    except SystemExit as e:
        assert e.code == 2


def test_missing_model_errors(capsys):
    try:
        main(["--engine", "ollama"])
        assert False, "Should have exited"
    except SystemExit as e:
        assert e.code == 2


# -- --extra flag tests -------------------------------------------------------


def test_extra_merges_into_engine_config():
    """--extra JSON is merged into EngineConfig.extra."""
    with (
        patch("runtimes_validator.cli.create_engine") as mock_create,
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "test-model",
                "--extra",
                '{"vllm_bin": "/usr/local/bin/vllm", "server_args": ["--device", "cpu"]}',
            ]
        )
        config = mock_create.call_args[0][1]
        assert config.extra["vllm_bin"] == "/usr/local/bin/vllm"
        assert config.extra["server_args"] == ["--device", "cpu"]


def test_extra_invalid_json_errors():
    """--extra with invalid JSON produces an error."""
    try:
        main(["--engine", "ollama", "--model", "m", "--extra", "{bad"])
        assert False, "Should have exited"
    except SystemExit as e:
        assert e.code == 2


def test_extra_non_object_errors():
    """--extra with a non-object JSON value produces an error."""
    try:
        main(["--engine", "ollama", "--model", "m", "--extra", '["a list"]'])
        assert False, "Should have exited"
    except SystemExit as e:
        assert e.code == 2


def test_extra_headers_take_precedence():
    """--header values override any 'headers' key in --extra."""
    with (
        patch("runtimes_validator.cli.create_engine") as mock_create,
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "m",
                "--extra",
                '{"headers": {"X-Old": "old"}}',
                "--header",
                "X-New: new",
            ]
        )
        config = mock_create.call_args[0][1]
        # --header overwrites the "headers" key from --extra
        assert config.extra["headers"] == {"X-New": "new"}


# -- --inspection-log extension warning --------------------------------------


def test_inspection_log_non_jsonl_extension_warns(tmp_path, capsys):
    """Passing a non-.jsonl path prints a warning on stderr but still runs."""
    log_path = tmp_path / "inspection.json"
    with (
        patch("runtimes_validator.cli.create_engine") as mock_create,
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "m",
                "--inspect",
                "--inspection-log",
                str(log_path),
            ]
        )

    err = capsys.readouterr().err
    assert "does not end in .jsonl" in err
    assert str(log_path) in err
    # logger was still created and wired up
    assert "inspection_logger" in mock_create.call_args[0][1].extra


def test_inspection_log_jsonl_extension_does_not_warn(tmp_path, capsys):
    """Passing a .jsonl path prints no warning."""
    log_path = tmp_path / "inspection.jsonl"
    with (
        patch("runtimes_validator.cli.create_engine"),
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "m",
                "--inspect",
                "--inspection-log",
                str(log_path),
            ]
        )

    err = capsys.readouterr().err
    assert "does not end in .jsonl" not in err


def test_inspection_log_without_inspect_errors(tmp_path):
    """--inspection-log requires --inspect."""
    log_path = tmp_path / "inspection.jsonl"
    try:
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "m",
                "--inspection-log",
                str(log_path),
            ]
        )
        assert False, "Should have exited"
    except SystemExit as e:
        assert e.code == 2


def test_inspection_log_uppercase_jsonl_extension_does_not_warn(tmp_path, capsys):
    """The extension check is case-insensitive."""
    log_path = tmp_path / "inspection.JSONL"
    with (
        patch("runtimes_validator.cli.create_engine"),
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "m",
                "--inspect",
                "--inspection-log",
                str(log_path),
            ]
        )

    err = capsys.readouterr().err
    assert "does not end in .jsonl" not in err
