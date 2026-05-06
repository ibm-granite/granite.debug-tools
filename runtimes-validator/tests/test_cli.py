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
