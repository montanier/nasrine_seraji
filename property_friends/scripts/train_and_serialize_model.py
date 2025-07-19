#!/usr/bin/env python3
"""Training script for property price prediction model."""

import click
from pathlib import Path

from property_friends.models import CATEGORICAL_COLS, TARGET, training, prediction
from property_friends.data_input.filesystem import load_data


@click.command()
@click.option(
    "--train-dataset-path",
    default="../data/datasets/train.csv",
    help="Path to training dataset CSV file",
    type=click.Path(exists=True),
)
@click.option(
    "--test-dataset-path",
    default="../data/datasets/test.csv",
    help="Path to test dataset CSV file",
    type=click.Path(exists=True),
)
@click.option(
    "--serialized-dir",
    default="../data/models/",
    help="Output path for serialized model",
    type=click.Path(path_type=Path),
)
def train(
    train_dataset_path: str, test_dataset_path: str, serialized_dir: Path
) -> None:
    """Train and serialize a property price prediction model."""
    click.echo("Starting training with:")
    click.echo(f"  Train dataset: {train_dataset_path}")
    click.echo(f"  Test dataset: {test_dataset_path}")
    click.echo(f"  Output dir: {serialized_dir}")

    click.echo("Load data")
    train_df, test_df = load_data(train_dataset_path, test_dataset_path)
    train_columns = list(train_df.columns)
    train_columns.remove(TARGET)
    train_inputs = train_df[train_columns]
    train_cat_inputs = train_df[CATEGORICAL_COLS]
    train_target = train_df[TARGET]

    test_cat_inputs = test_df[CATEGORICAL_COLS]
    test_inputs = test_df[train_columns]
    test_target = test_df[TARGET]

    click.echo("Fit preprocessor")
    pre_processor = training.get_fitted_preprocessor(train_cat_inputs, train_target)

    click.echo("Transform dataset")
    train_cat_features = training.get_transformed_dataset(
        train_cat_inputs, pre_processor
    )
    train_inputs.loc[:, CATEGORICAL_COLS] = train_cat_features.values

    test_cat_features = training.get_transformed_dataset(test_cat_inputs, pre_processor)
    test_inputs.loc[:, CATEGORICAL_COLS] = test_cat_features.values

    click.echo("Train model")
    model = training.get_trained_model(train_inputs, train_target)

    click.echo("Evaluate model on test")
    predictions = model.predict(test_inputs)
    metrics = training.get_metrics(predictions, test_target.values)
    click.echo(f"   {metrics}")

    click.echo("Serialize result")
    serialized_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = Path(serialized_dir) / "preprocessor.joblib"
    training.serialize_preprocessor(pre_processor, preprocessor_path)
    model_path = Path(serialized_dir) / "model.joblib"
    training.serialize_preprocessor(model, model_path)


if __name__ == "__main__":
    train()
