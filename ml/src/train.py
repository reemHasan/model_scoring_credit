
from sklearn.metrics import confusion_matrix
# import required modules
import pandas as pd
from functions import calcul_metric, plot_confusion_matrix
import joblib
# Modeling
import lightgbm as lgb
# mlflow
import mlflow
from mlflow import MlflowClient
import mlflow.data
import os

if not os.path.exists("../artifacts/"):
    os.makedirs("../artifacts/")

# Load the training and test datasets
train_data = pd.read_parquet('../../data/training_data/train_data_bestFeatures.parquet')
test_data = pd.read_parquet('../../data/training_data/test_data_bestFeatures.parquet')
print("Train dataset shape: ",train_data.shape)
print("Test dataset shape: ",test_data.shape)
# Load the best model bundle from the grid search
bundle = joblib.load('../model/grid_fbeta_model_bundle.pkl')
all_params  = bundle['all_params']
features   = bundle['feature_names']
best_threshold = bundle['threshold_fbeta10']
# Initialize MLflow client and set the experiment
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
projet1_exp1 = mlflow.set_experiment("credit_scoring_experiment")
# Train the best LGBM classifier and log the model, metrics, and parameters to MLflow
with mlflow.start_run(run_name="Best LGBMclassifier fbeta10") as run:
    lgbm = lgb.LGBMClassifier(**all_params)
    lgbm.fit(train_data[features],train_data['Target'])
    # Log model
    model_info_lgb = mlflow.lightgbm.log_model(lgb_model=lgbm, name="Best LGBMclassifier for HomeCreditRisk")
    # Predict on the train set, compute and log the metrics
    y_pred_proba = lgbm.predict_proba(train_data[features])[:, 1]               #get probablity
    y_pred = lgbm.predict(train_data[features])
    t_roc_auc, t_f1, t_accuracy, t_fbeta = calcul_metric(train_data['Target'], y_pred, y_pred_proba)
    print(f"LGBM classifier on Train : ROC-AUC = {t_roc_auc:.4f}")
    print(f"LGBM classifier on Train : F1 score = {t_f1:.4f}")
    print(f"LGBM classifier on Train  Accuracy: {t_accuracy:.4f}")
    print(f"LGBM classifier on Train  fbeta score: {t_fbeta:.4f}")
    # Predict on the test set, compute and log the metrics
    y_test_pred_proba = lgbm.predict_proba(test_data[features])[:, 1]               #get probablity
    y_test_pred = lgbm.predict(test_data[features])
    v_roc_auc, v_f1, v_accuracy, v_fbeta = calcul_metric(test_data['Target'], y_test_pred, y_test_pred_proba)
    print(f"LGBM classifier on test: ROC-AUC = {v_roc_auc:.4f}")
    print(f"LGBM classifier on test: F1 score = {v_f1:.4f}")
    print(f"LGBM classifier on test Accuracy: {v_accuracy:.4f}")
    print(f"LGBM classifier on test fbeta score: {v_fbeta:.4f}")
    mlflow.log_metrics({"Train Roc-Auc score": t_roc_auc,
                        "Train F1 score": t_f1,
                        "Train Fbeta score":t_fbeta,
                        "Train accuracy": t_accuracy})
    mlflow.log_metrics({"Validate Roc-Auc score": v_roc_auc,
                        "Validate F1 score": v_f1,
                        "Validate Fbeta score":v_fbeta,
                        "Validate accuracy": v_accuracy})
    cm = confusion_matrix(test_data['Target'], y_test_pred)
    plot_confusion_matrix(cm,["Repay loan","Not Repay loan"],"cm_lgbm_bestModel", image_path="../artifacts/")
    mlflow.log_artifact("../artifacts/cm_lgbm_bestModel.png")
    # Log the hyperparameters
    mlflow.log_params(lgbm.get_params())
    mlflow.set_tag("Training Info", "Best LGBMclassifier for HomeCreditRisk data with roc=0.75")

# Bundle model with useful metadata
model_bundle = {
    'model':          lgbm,
    'feature_names':  features,
    'threshold': best_threshold,
    'Target_column': 'Target',
    'target_names':   ['Repay_loan', 'Not_repay_loan'],
    'all_params':     lgbm.get_params(),
    'metrics': {
        'train_roc':   t_roc_auc,
        'test_roc':    v_roc_auc,
    }
}

# Save the model bundle to a file
joblib.dump(model_bundle, '../model/lgbm_bestmodel_fbeta10_bundle.pkl', compress=3)
print("Model bundle saved successfully!")