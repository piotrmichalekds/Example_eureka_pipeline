from eurekaai.components.steps import step, Output
from eurekaai.components.pipelines import pipeline
import pandas as pd


@step(enable_cache=False)
def generate_data() -> Output(
    df=pd.DataFrame,
    x_train=pd.DataFrame,
    x_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series
    ):
    import pandas as pd
    import numpy as np
    import random
    from datetime import datetime, timedelta
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
        
    selected_feats = [f"feature_{i}" for i in range(60)]
    # Parameters for make_classification
    n_samples = 1000
    n_features = len(selected_feats)
    n_informative = 18
    n_redundant = 9
    n_classes = 2

    # Generate synthetic classification dataset
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_classes=n_classes, random_state=42)

    # Convert to DataFrame
    df_features = pd.DataFrame(X, columns=selected_feats)
    df_labels = pd.DataFrame(y, columns=['ml_label'])

    # Meta columns
    meta_cols = ['primary_customer_id', 'featuredate']
    columns = meta_cols + selected_feats + ['ml_label']

    # Generate random meta data
    meta_data = {
        'primary_customer_id': [f'cust_{i}' for i in range(n_samples)],
        'featuredate': [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(n_samples)]
    }

    # Create the final DataFrame
    df_meta = pd.DataFrame(meta_data)
    df = pd.concat([df_meta, df_features, df_labels], axis=1)

    df = df.replace(np.inf, 1e33)
    df = df.replace(-np.inf, -1e33).fillna(0)
    
    x, y = df[selected_feats], df['ml_label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return df, x_train, x_test, y_train, y_test

@step(enable_cache=False)
def lf_extraction(df:pd.DataFrame,
    x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series) -> Output(
    extracted_lf=dict):
    
    import pandas as pd
    from senpy.api.lf_extraction import LFExtraction
    import os
    import senpy.api.model_manager as mm
    from senpy.utils.misc import get_logger
    
    #set admin creds
    with open('env_config.json', 'r') as f: env_config = json.load(f)

    os.environ['MLFLOW_ADMIN_USERNAME'] = env_config['MLFLOW_ADMIN_USERNAME'] 
    os.environ['MLFLOW_ADMIN_PASSWORD'] = env_config['MLFLOW_ADMIN_PASSWORD'] 
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = env_config['AZURE_STORAGE_CONNECTION_STRING'] 
    os.environ['AZURE_CONTAINER'] = env_config['AZURE_CONTAINER']
    logger = get_logger()
    
    lf_extraction_mm = mm.ModelManager(
        model_name="test_lf_model_azure1",
        pipeline_family="test",
        mlflow_tracking_uri="https://mlflow-eureka-dev.fsai-symphonyai.com/",
        mlflow_username="username",
        mlflow_password="password",
        azure_connection_string=os.environ.get('AZURE_STORAGE_CONNECTION_STRING'),
        azure_container=os.environ.get('AZURE_CONTAINER'),
        logger=logger
    )
    
    GEN = 'gen2'
    SOURCE_UNIT = 'suncorp'
    CFG_PATH = f"configs/{GEN}/{SOURCE_UNIT}"
    FEAT = 'united'
    FEAT_DIV = ''  # values=('', '_all', '_nrm', '_qnt')
    
    config = {"group": "null",
    "filename": "suncorp_numleaves_1024",
    "folder": "export/gen2/suncorp",
    "stamp": True,
    "pre_filter": {"min_0_samples_class_0": 1,
                "max_1_samples_class_0": 10000,
                "min_1_samples_class_1": 1,
                "max_0_samples_class_1": 10000,
                "acc_class_0": 0.4,
                "acc_class_1": 0.4,
                "max_rules_class_0": 10000,
                "max_rules_class_1": 10000
               },
    "class_0": {"acc_threshold": 0.7,
             "max_incorrect": 1000,
             "overlap_limit": 95,
             "rules_to_cover": 3,
             "data_coverage": 99,
             "rules_limit": 1000,
             "penalty_factor": 2,
             "weight_gini_impurity": 0.4,
             "weight_split_ratio": 0.2,
             "weight_adjusted_accuracy": 0.4
             },
    "class_1": {"acc_threshold": 0.8,
             "max_incorrect": 200,
             "overlap_limit": 99,
             "rules_to_cover": 4,
             "data_coverage": 99,
             "rules_limit": 3000,
             "penalty_factor": 2,
             "weight_gini_impurity": 0.3,
             "weight_split_ratio": 0.5,
             "weight_adjusted_accuracy": 0.2
            }
    }

    params = {'n_estimators': 300,
      'max_leaf_nodes': None,
      'min_samples_leaf': 2,
      'random_state': 27,
      'n_jobs': 8,
      'max_features': None,
      'max_depth': 11}

    opt_params = {'name': 'max_leaf_nodes',
              'init_value': 512,
              'step': 512}
    bu_id = 'a18a807e-9a48-4edb-b7fb-cb2d66d89047'
    #end training run
    lf_extraction_mm.end_training_run()
    # start training run
    training_session = lf_extraction_mm.start_training_run()
    lf_extraction_mm.log_dataset(df, 'dataset_name', 'train')

    # Initialize LFExtraction
    lf_extraction = LFExtraction((x_train, y_train), config, params=params)

    # generate label functions
    lf_extraction.rules_generation_and_extraction(opt_params)
    lf_extraction.pre_filtering_rules()
    lf_extraction.build_label_functions(FEAT, FEAT_DIV)
    lf_extraction.label_function_evaluation()
    lf_extraction.label_function_selection()

    # # Save model parameters
    lf_extraction.log_model_params_and_config()
    lf_extraction.plot_rules()
    lf_extraction.log_plots()
    lf_extraction.log_payload(bu_id)
    lf_extraction.log_metrics((x_test, y_test))
    lf_extraction.log_coverage_plot()

    # lf_extraction.register_lf(base_url, email, password)

    lf_extraction_mm.end_training_run()
    extracted_lf = lf_extraction.payload

    return extracted_lf

@step(enable_cache=False)
def lf_model(df:pd.DataFrame,
    x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.Series,
    extracted_lf:dict) -> None:
    
    import pandas as pd
    from senpy.api.lf_model import LFModel
    import os
    import senpy.api.model_manager as mm
    from senpy.utils.misc import get_logger
    from senpy.source.knn import DenoiserKNN
    from senpy.source.majority import DenoiserMajority
    import logging
    logging.getLogger("mlflow").setLevel(logging.DEBUG)
    
    def knn_denoiser():
        knn_index = 0
        knn_value = [2, 3, 5, 7, 9]
        knn_algorithm = ['ball_tree', 'kd_tree', 'brute']
        knn_metric = ['l2', 'l1', 'infinity', 'cosine']
        knn_params = {"algorithm": knn_algorithm[knn_index], "metric": knn_metric[knn_index]}
        knn_denoiser = DenoiserKNN(knn=knn_value[knn_index],
                                    resolve_tie=False,
                                    probs=False,
                                    parallel=True,
                                    n_jobs=4,
                                    **knn_params)
        return knn_denoiser

    def base_denoiser():
        base_denoiser = DenoiserMajority(probs=False, relosve_tie=False)
        return base_denoiser
    
    #set admin creds
    with open('env_config.json', 'r') as f: env_config = json.load(f)

    os.environ['MLFLOW_ADMIN_USERNAME'] = env_config['MLFLOW_ADMIN_USERNAME'] 
    os.environ['MLFLOW_ADMIN_PASSWORD'] = env_config['MLFLOW_ADMIN_PASSWORD'] 
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = env_config['AZURE_STORAGE_CONNECTION_STRING'] 
    os.environ['AZURE_CONTAINER'] = env_config['AZURE_CONTAINER']
    logger = get_logger()
    
    lf_model_mm = mm.ModelManager(
        model_name="test_lf_model_azure1",
        pipeline_family="test",
        mlflow_tracking_uri="https://mlflow-eureka-dev.fsai-symphonyai.com/",
        mlflow_username="username",
        mlflow_password="password",
        azure_connection_string=os.environ.get('AZURE_STORAGE_CONNECTION_STRING'),
        azure_container=os.environ.get('AZURE_CONTAINER'),
        logger=logger
        )
    
    lf_model = LFModel(extracted_lf)    
    lf_model.fit(x_train, y_train, knn_denoiser())    
    predictions = lf_model.predict(x_test)
    
    lf_model_mm.end_training_run()

    training_session = lf_model_mm.start_training_run()

    lf_model_mm.log_dataset(df, 'dataset_name', 'dataset_type')

    predictions = lf_model.predict(x_test)
    lf_model_mm.save_model(lf_model, x_test, predictions, predictor_class=mm.LFModelPredictor)

    lf_model_mm.end_training_run()

    
@pipeline(enable_cache=False, settings={"container": container_settings, "kubeflow": kubeflow_setting})
def label_functions_pipeline_v1(
     generate_data,
     lf_extraction,
     lf_model
    ):
    df, x_train, x_test, y_train, y_test = generate_data()
    extracted_lf = lf_extraction(df, x_train, x_test, y_train, y_test)
    lf_model(df, x_train, x_test, y_train, extracted_lf)
