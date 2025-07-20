import pytest
import pandas as pd
import tempfile
import os
from property_friends.data_input.filesystem import load_data, FileSystemDataLoader
from property_friends.data_input.factory import create_data_loader


def test_load_data() -> None:
    # Given
    train_data = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [10, 20, 30]}
    )

    test_data = pd.DataFrame({"feature1": [4, 5], "feature2": ["d", "e"]})

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # When
        loaded_train, loaded_test = load_data(train_path, test_path)

        # Then
        pd.testing.assert_frame_equal(loaded_train, train_data)
        pd.testing.assert_frame_equal(loaded_test, test_data)


def test_load_data_file_not_found() -> None:
    # Given
    nonexistent_train_path = "nonexistent_train.csv"
    nonexistent_test_path = "nonexistent_test.csv"

    # When / Then
    with pytest.raises(FileNotFoundError):
        load_data(nonexistent_train_path, nonexistent_test_path)


def test_filesystem_data_loader() -> None:
    # Given
    train_data = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [10, 20, 30]}
    )
    test_data = pd.DataFrame({"feature1": [4, 5], "feature2": ["d", "e"]})

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # When
        loader = FileSystemDataLoader(train_path, test_path)
        loaded_train = loader.load_training_data()
        loaded_test = loader.load_test_data()

        # Then
        pd.testing.assert_frame_equal(loaded_train, train_data)
        pd.testing.assert_frame_equal(loaded_test, test_data)


def test_filesystem_data_loader_load_data() -> None:
    # Given
    train_data = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [10, 20, 30]}
    )
    test_data = pd.DataFrame({"feature1": [4, 5], "feature2": ["d", "e"]})

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # When
        loader = FileSystemDataLoader(train_path, test_path)
        loaded_train, loaded_test = loader.load_data()

        # Then
        pd.testing.assert_frame_equal(loaded_train, train_data)
        pd.testing.assert_frame_equal(loaded_test, test_data)


def test_create_data_loader_filesystem() -> None:
    # Given
    train_data = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": ["a", "b", "c"], "target": [10, 20, 30]}
    )
    test_data = pd.DataFrame({"feature1": [4, 5], "feature2": ["d", "e"]})

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        test_path = os.path.join(temp_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # When
        loader = create_data_loader(
            "filesystem", train_path=train_path, test_path=test_path
        )
        loaded_train, loaded_test = loader.load_data()

        # Then
        assert isinstance(loader, FileSystemDataLoader)
        pd.testing.assert_frame_equal(loaded_train, train_data)
        pd.testing.assert_frame_equal(loaded_test, test_data)


def test_create_data_loader_unknown_type() -> None:
    # When / Then
    with pytest.raises(ValueError, match="Unknown loader type: unknown"):
        create_data_loader("unknown", some_param="value")


def test_create_data_loader_missing_params() -> None:
    # When / Then
    with pytest.raises(ValueError, match="Missing required parameters"):
        create_data_loader("filesystem", train_path="path.csv")  # Missing test_path


def test_filesystem_data_loader_file_not_found() -> None:
    # Given
    loader = FileSystemDataLoader("nonexistent_train.csv", "nonexistent_test.csv")

    # When / Then
    with pytest.raises(FileNotFoundError):
        loader.load_training_data()

    with pytest.raises(FileNotFoundError):
        loader.load_test_data()
