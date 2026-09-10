
from pathlib import Path


# File types the system is currently allowed to inspect.
SUPPORTED_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".md",
    ".txt",
}


# Directories that should not be searched.
IGNORED_DIRECTORIES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}


def list_project_files(project_path: str) -> list[str]:
    """
    Return supported files found inside a project directory.

    The returned paths are relative to the project root. Ignored directories
    are skipped, and no project files are modified.
    """

    # This converts the supplied string into a Path object and converts it to an absolute path.
    # root now represents the project’s main folder.
    root = Path(project_path).resolve()

    # This checks whether the path actually exists. The function expects the main project folder,
    if not root.exists():
        raise FileNotFoundError(
            f"Project path does not exist: {project_path}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Project path is not a directory: {project_path}"
        )

    files = []

    # This searches recursively through everything inside the project.
    # it searches both the main folder and all nested subfolders.
    for path in root.rglob("*"):
        # This removes the project’s absolute root from the path.
        relative_path = path.relative_to(root)

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative_path.parts
        ):
            # checks whether any folder or filename component in the path is an ignored directory.
            # Skip this path and move to the next item in the for loop.
            continue

        if (path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES):
            # The path must be a file, not a directory
            # extension must be supported.lower() handles uppercase extensions
            
            # If the file passes all checks, its relative path is converted to a string and added to the list.
            files.append(str(relative_path))

    return sorted(files)



def read_project_file(
    project_path: str,
    relative_file_path: str,
) -> str:
    """
    Read and return the text content of one supported project file.

    The file must be inside the selected project directory.
    This function reads the file but does not modify it.
    """

    # Convert the project folder path into an absolute Path object.
    root = Path(project_path).resolve()

    # Create the complete path to the requested file.
    file_path = (root / relative_file_path).resolve()

    # Make sure the requested file is still inside the project folder.
    # This prevents access to outside files using paths such as "../file.txt".
    try:
        file_path.relative_to(root)
    except ValueError as error:
        raise PermissionError(
            "Cannot read files outside the project directory."
        ) from error

    # Check that the requested path exists.
    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {relative_file_path}"
        )

    # Check that the requested path is a file and not a directory.
    if not file_path.is_file():
        raise IsADirectoryError(
            f"Path is not a file: {relative_file_path}"
        )

    # Check that the file type is supported.
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    # Read and return the complete text content of the file.
    return file_path.read_text(encoding="utf-8")