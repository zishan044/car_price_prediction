from kedro.pipeline import Node, Pipeline
from .nodes import clean_data, extract_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=clean_data, inputs='cars', outputs='cleaned_cars', name='clean_data_node'),
        Node(func=extract_features, inputs='cleaned_cars', outputs='feature_data', name='extract_features_node')
    ])

