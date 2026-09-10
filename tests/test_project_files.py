from pathlib import Path

import pytest

from tools.project_files import list_project_files, read_project_file


@pytest.mark.unit
def test_project_listing_filters_files(tmp_path):
    (tmp_path / "model.py").write_text("model", encoding="utf-8")
    (tmp_path / "weights.bin").write_bytes(b"weights")

    source_directory = tmp_path / "src"
    source_directory.mkdir()
    (source_directory / "train.py").write_text(
        "training code",
        encoding="utf-8",
    )

    ignored_directory = tmp_path / ".venv"
    ignored_directory.mkdir()
    (ignored_directory / "hidden.py").write_text(
        "ignored code",
        encoding="utf-8",
    )

    results = list_project_files(str(tmp_path))

    assert results == sorted([
        "model.py",
        str(Path("src") / "train.py"),
    ])


@pytest.mark.unit
def test_existing_file_returns_content(tmp_path):
    content = "learning_rate = 0.01\n"
    (tmp_path / "train.py").write_text(content, encoding="utf-8")

    result = read_project_file(
        project_path=str(tmp_path),
        relative_file_path="train.py",
    )

    assert result == content


@pytest.mark.unit
def test_outside_file_raises(tmp_path):
    project_directory = tmp_path / "project"
    project_directory.mkdir()

    (tmp_path / "outside.txt").write_text(
        "Outside project",
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="outside the project directory"):
        read_project_file(
            project_path=str(project_directory),
            relative_file_path="../outside.txt",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "path_name, expected_error",
    [
        ("missing", FileNotFoundError),
        ("file.txt", NotADirectoryError),
    ],
)
def test_invalid_project_raises(tmp_path, path_name, expected_error):
    (tmp_path / "file.txt").write_text("test", encoding="utf-8")

    with pytest.raises(expected_error):
        list_project_files(str(tmp_path / path_name))


@pytest.mark.unit
@pytest.mark.parametrize(
    "path_name, expected_error",
    [
        ("missing.py", FileNotFoundError),
        ("folder.py", IsADirectoryError),
        ("weights.bin", ValueError),
    ],
)
def test_invalid_file_raises(tmp_path, path_name, expected_error):
    (tmp_path / "folder.py").mkdir()
    (tmp_path / "weights.bin").write_bytes(b"weights")

    with pytest.raises(expected_error):
        read_project_file(
            project_path=str(tmp_path),
            relative_file_path=path_name,
        )