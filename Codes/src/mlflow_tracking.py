import os
import pickle
import pandas as pd
import numpy as np
from LG import *
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
nltk.download('stopwords')
nltk.download('punkt')
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "masoud"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Strong#Pass#2022"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
# Force MLflow à enregistrer les runs localement dans un dossier au lieu de chercher un serveur
mlflow.set_tracking_uri("file:///var/jenkins_home/workspace/spam-detection-pipeline/mlruns")
mlflow.set_experiment("Spam_Detection")
#model
models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'SVM': SVC(kernel='linear', class_weight='balanced', probability=True),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
}


#category model
for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ('clf', clf),
        ])
        # Entraînement
        pipeline.fit(X_train, y_train)
        # Prédiction sur le jeu de test
        y_pred = pipeline.predict(X_test)
        # Calcul des métriques de classification
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='Spam', zero_division=0) 
        rec = recall_score(y_test, y_pred, pos_label='Spam', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='Spam', zero_division=0)
        # Envoi à MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)  # Métrique critique soulignée par le notebook !
        mlflow.log_metric("f1_score", f1)
        # Sauvegarde textuelle du rapport dans les artefacts de MLflow
        report = classification_report(y_test, y_pred)
        with open(f"report_{name}.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact(f"report_{name}.txt")
        if os.path.exists("Figure/scatter.png"):
            mlflow.log_artifact("Figure/scatter.png")
        # Sauvegarde du modèle dans MLflow
        mlflow.sklearn.log_model(
            pipeline, 
            artifact_path=f"model_{name}",
            skops_trusted_types=["scipy.sparse._csr.csr_matrix"]
        )
        print(f"Modèle {name} enregistré avec succès. (Recall: {rec:.4f})")
        print(classification_report(y_test, y_pred))

#optimisation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
#Grid Search
with mlflow.start_run(run_name="LogisticRegression_GridSearch"):    
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__class_weight': [None, 'balanced'],
    }
    # Pas de TfidfVectorizer ici car X_train est déjà numérique !
    pipeline_grid = Pipeline([
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear')),
    ])
    # On utilise pos_label='Spam' pour correspondre à tes données
    scorer = make_scorer(recall_score, pos_label='Spam')
    grid = GridSearchCV(pipeline_grid, param_grid, scoring=scorer, cv=5)
    grid.fit(X_train, y_train)
    # 2. On envoie les meilleurs paramètres trouvés à MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_recall_score", grid.best_score_)
    # 3. Sauvegarde du meilleur modèle trouvé
    mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="best_grid_model")
    print("Meilleurs paramètres :", grid.best_params_)
    print(f"Meilleur score de Recall : {grid.best_score_:.4f}")

#application sur dataset complet
X_full = pd.concat([X_train_text, X_test_text])
y_full = pd.concat([y_train, y_test])
#creation dossier auto
models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)
#creation pipeline du top model
final_model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(C=1, class_weight='balanced', max_iter=1000)) # Tes meilleurs paramètres
])
#entrainement
final_model.fit(X_full, y_full)
#save
model_path = os.path.join(models_dir, "logreg_spam_pipeline.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)

#holdout set 
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
    data['mess_clean'], 
    data['Category'], 
    test_size=0.10,      # On garde 10% strictly pour le test final de l'oral
    stratify=data['Category'], 
    random_state=42
)
#chargement top model
model_path = os.path.join("..", "models", "logreg_spam_pipeline.pkl")
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_pipeline = pickle.load(f)
    #prediction
    y_pred = model_pipeline.predict(X_holdout)
    #rapport
    print("\nRapport de classification final sur 10%:")
    print(classification_report(y_holdout, y_pred))   
else:
    print("le model n'existe pas")
    
print('Mlflow Tracking mode activer')
