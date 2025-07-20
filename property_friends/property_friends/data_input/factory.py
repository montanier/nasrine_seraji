"""Factory for creating data loaders."""

from typing import Any, Dict
from .base import DataLoader
from .filesystem import FileSystemDataLoader


def create_data_loader(loader_type: str, **kwargs: Any) -> DataLoader:
    """Factory function to create data loaders.

    Args:
        loader_type: Type of loader ('filesystem', 'database', etc.)
        **kwargs: Loader-specific configuration parameters

    Returns:
        Configured DataLoader instance.

    Raises:
        ValueError: If loader_type is not supported.

    Example:
        >>> loader = create_data_loader(
        ...     "filesystem",
        ...     train_path="data/train.csv",
        ...     test_path="data/test.csv"
        ... )
        >>> train_df, test_df = loader.load_data()
    """
    if loader_type == "filesystem":
        required_params = ["train_path", "test_path"]
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for filesystem loader: {missing_params}"
            )

        return FileSystemDataLoader(
            train_path=kwargs["train_path"], test_path=kwargs["test_path"]
        )
    # Future implementations:
    # elif loader_type == "database":
    #     return DatabaseDataLoader(kwargs["connection_string"])
    # elif loader_type == "cloud":
    #     return CloudDataLoader(kwargs["bucket"], kwargs["prefix"])
    else:
        raise ValueError(
            f"Unknown loader type: {loader_type}. Supported types: filesystem"
        )
