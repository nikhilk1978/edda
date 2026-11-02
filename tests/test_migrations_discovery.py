"""Migration discovery must work from any working directory (editable installs)."""

from edda.storage.migrations import find_migrations_dir


def test_find_migrations_dir_does_not_depend_on_cwd(tmp_path, monkeypatch):
    # A consuming project has no ``schema/`` of its own. Before the checkout-relative
    # lookup, an editable install found no migrations here and silently created no tables.
    monkeypatch.chdir(tmp_path)

    found = find_migrations_dir()

    assert found is not None
    assert (found / "sqlite").is_dir()
