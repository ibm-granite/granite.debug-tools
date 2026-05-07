from unittest.mock import patch

from runtimes_validator.cli import main


def _run_cli(extra_args):
    args = ["--engine", "ollama", "--model", "m", *extra_args]
    with (
        patch("runtimes_validator.cli.create_engine"),
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
        patch("runtimes_validator.cli.get_tests") as mock_get_tests,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        mock_get_tests.return_value = []
        rc = main(args)
        return rc, mock_get_tests


def test_modalities_default_is_text():
    rc, mock_get_tests = _run_cli([])
    assert rc == 0
    mock_get_tests.assert_called_once_with(engine_id="ollama", modalities={"text"})


def test_modalities_single_value():
    rc, mock_get_tests = _run_cli(["--modalities", "vision"])
    assert rc == 0
    mock_get_tests.assert_called_once_with(engine_id="ollama", modalities={"vision"})


def test_modalities_multiple_values():
    rc, mock_get_tests = _run_cli(["--modalities", "vision,speech"])
    assert rc == 0
    mock_get_tests.assert_called_once_with(engine_id="ollama", modalities={"vision", "speech"})


def test_modalities_ignored_when_tests_is_explicit():
    """With --tests set, get_tests is not called and explicit IDs win."""
    with (
        patch("runtimes_validator.cli.create_engine"),
        patch("runtimes_validator.cli.ValidationRunner") as mock_runner,
        patch("runtimes_validator.cli.get_tests") as mock_get_tests,
        patch("runtimes_validator.cli.get_test_by_id") as mock_by_id,
    ):
        mock_runner.return_value.run.return_value.all_passed = True
        main(
            [
                "--engine",
                "ollama",
                "--model",
                "m",
                "--tests",
                "basic_generation",
                "--modalities",
                "vision",
            ]
        )
        mock_get_tests.assert_not_called()
        mock_by_id.assert_called_with("basic_generation")


def test_list_tests_shows_modalities(capsys):
    rc = main(["--list-tests"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "basic_generation\t[text]" in output
    assert "vision_basic\t[vision]" in output
    assert "speech_basic\t[speech]" in output
