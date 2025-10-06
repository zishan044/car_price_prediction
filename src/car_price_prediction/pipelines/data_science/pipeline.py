from kedro.pipeline import Pipeline, Node
from .nodes import split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=split_data,
            inputs=['feature_data', 'params:active_modeling_pipeline.model_options'],
            outputs=['X_train', 'X_test', 'y_train', 'y_test'],
            name='split_data_node',
        ),
        Node(
            func=train_model,
            inputs=['X_train', 'y_train'],
            outputs='gradient_boosting_regressor',
            name='train_model_node',
        ),
        Node(
            func=evaluate_model,
            inputs=['gradient_boosting_regressor', 'X_test', 'y_test'],
            outputs=None,
            name='evaluate_model_node',
        ),
    ])