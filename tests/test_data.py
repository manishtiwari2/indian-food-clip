from conftest import CLASSES, PER_CLASS, ROOT

from foodclip.data import (
    dedup, find_class_root, is_valid_image, list_images, load_class_names, stratified_split,
)


def test_find_class_root_walks_nested_folders(fake_indian):
    assert find_class_root(fake_indian).name == "Indian Food Images"
    assert {d.name for d in find_class_root(fake_indian).iterdir()} == set(CLASSES)


def test_corrupt_and_duplicate_are_removed(fake_indian):
    class_root, df = list_images(fake_indian)
    assert len(df) == len(CLASSES) * PER_CLASS + 2
    valid = df[[is_valid_image(class_root / r) for r in df["relpath"]]]
    assert "jalebi/broken.jpg" not in set(valid["relpath"])
    kept, removed = dedup(valid, class_root, max_distance=0)
    assert len(kept) == len(CLASSES) * PER_CLASS
    assert set(removed["relpath"]) & {"poha/000.jpg", "poha/dup.jpg"}
    assert (removed["reason"] == "duplicate").all()


def test_cross_class_duplicate_drops_both_copies(fake_indian):
    import shutil

    class_root = find_class_root(fake_indian)
    shutil.copy(class_root / "jalebi" / "001.jpg", class_root / "rasgulla" / "copy.jpg")
    _, df = list_images(class_root)
    df = df[[is_valid_image(class_root / r) for r in df["relpath"]]]
    kept, removed = dedup(df, class_root, max_distance=0)
    cross = removed[removed.reason == "cross_class_duplicate"]
    assert set(cross["relpath"]) == {"jalebi/001.jpg", "rasgulla/copy.jpg"}
    assert not set(cross["relpath"]) & set(kept["relpath"])
    assert len(kept) == len(CLASSES) * PER_CLASS - 1  # jalebi/001 gone, copy gone, poha dup gone


def test_split_has_no_leakage_and_is_deterministic(fake_indian):
    _, df = list_images(fake_indian)
    a = stratified_split(df, test_per_class=3, seed=42)
    b = stratified_split(df, test_per_class=3, seed=42)
    assert a.equals(b)
    train = set(a[a.split == "train"].relpath)
    test = set(a[a.split == "test"].relpath)
    assert not train & test
    assert (a[a.split == "test"].groupby("label").size() == 3).all()


def test_small_class_still_gets_test_images(fake_indian):
    _, df = list_images(fake_indian)
    split = stratified_split(df, test_per_class=15, seed=0)  # 12 imgs/class < 2*15
    assert (split[split.split == "test"].groupby("label").size() >= 1).all()
    assert (split[split.split == "train"].groupby("label").size() >= 1).all()


def test_class_names_csv_is_complete():
    names = load_class_names(ROOT / "data" / "class_names.csv")
    assert len(names) == 80
    assert names["folder"].is_unique
    assert not names[["english", "hindi_roman", "hindi_devanagari"]].isna().any().any()
    # every Devanagari entry actually contains Devanagari characters
    assert names["hindi_devanagari"].map(lambda s: any("ऀ" <= ch <= "ॿ" for ch in s)).all()
