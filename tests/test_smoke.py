from src.database.sqlite_utils import get_database_url, get_project_status, init_database


def test_database_initialization_and_status():
    init_database()
    assert get_database_url().startswith("sqlite:///")
    assert isinstance(get_project_status(), str)
    assert get_project_status()
