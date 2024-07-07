from mage_ai.data_prep.decorators import pipeline, python_file, data_loader, transformer, model
from mage_ai.data_prep.repo_service import get_repo_service
from mage_ai.data_prep.variable_manager import get_variable_manager

@pipeline(
    name='attrition-prediction',
    python_file='attrition-prediction.py',
    entry_point='main',
)
def main():
    repo_service = get_repo_service()
    variable_manager = get_variable_manager()

    load_and_clean_data = data_loader('load_and_clean_data', python_file='transformers/data_cleaning.py')
    select_features = transformer('select_features', python_file='features/feature_selection.py', inputs={'data': load_and_clean_data}, arguments={'selected_features': variable_manager.get_variable('selected_features')})
    preprocess_data = transformer('preprocess_data', python_file='features/preprocessing.py', inputs={'data': load_and_clean_data})
    train_and_evaluate_models = model('train_and_evaluate_models', python_file='models/training.py', inputs={
        'X_train': preprocess_data.outputs['X_train'],
        'X_val': preprocess_data.outputs['X_val'],
        'X_test': preprocess_data.outputs['X_test'],
        'y_train': preprocess_data.outputs['y_train'],
        'y_val': preprocess_data.outputs['y_val'],
        'y_test': preprocess_data.outputs['y_test'],
    })

    repo_service.run_pipeline(
        pipeline=main,
        variables={
            'selected_features': [
                'DailyRate',
                'DistanceFromHome',
                'EnvironmentSatisfaction',
                'JobInvolvement',
                'JobSatisfaction',
                'MaritalStatus',
                'NumCompaniesWorked',
                'OverTime',
                'StockOptionLevel',
                'TotalWorkingYears',
                'TrainingTimesLastYear',
                'WorkLifeBalance',
                'YearsAtCompany',
            ],
        },
    )

if __name__ == '__main__':
    main()