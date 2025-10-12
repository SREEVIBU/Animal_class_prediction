import os
import argparse
import json
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, jsonify

# Paths
BASE_DIR = os.path.dirname(__file__)
ZOO_CSV = os.path.join(BASE_DIR, "zoo.csv")
CLASS_CSV = os.path.join(BASE_DIR, "class.csv")
MODEL_PATH = os.path.join(BASE_DIR, "zoo_rf_model.joblib")


def load_data(zoo_path: str = ZOO_CSV) -> pd.DataFrame:
    """Load zoo dataset."""
    df = pd.read_csv(zoo_path)
    return df


def build_label_mapping(class_csv: str = CLASS_CSV) -> Dict[int, str]:
    """Return mapping from class_type integer to human readable class name.

    The `class.csv` file uses `Class_Number` as id and `Class_Type` as name.
    The `zoo.csv` file uses `class_type` column with integer values.
    """
    if not os.path.exists(class_csv):
        return {}
    df = pd.read_csv(class_csv)
    mapping = {}
    # Try to read rows, fallback to possible different column names
    if 'Class_Number' in df.columns and 'Class_Type' in df.columns:
        for _, r in df.iterrows():
            mapping[int(r['Class_Number'])] = str(r['Class_Type'])
    else:
        # last resort: map class_type value to Class_Type if present
        for _, r in df.iterrows():
            keys = [c for c in df.columns if 'class' in c.lower() or 'number' in c.lower()]
            vals = [c for c in df.columns if 'type' in c.lower() or 'class' in c.lower()]
            if keys and vals:
                try:
                    mapping[int(r[keys[0]])] = str(r[vals[-1]])
                except Exception:
                    continue
    return mapping


def train_and_save(zoo_csv: str = ZOO_CSV, model_path: str = MODEL_PATH) -> Dict:
    df = load_data(zoo_csv)
    # The dataset contains 'animal_name' and many binary features plus 'class_type'
    if 'class_type' not in df.columns:
        raise ValueError("zoo.csv must contain a 'class_type' column")

    X = df.drop(columns=['animal_name', 'class_type'])
    y = df['class_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    joblib.dump(clf, model_path)

    return {"accuracy": acc, "report": report, "model_path": model_path}


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_from_features(features: Dict[str, int], model) -> int:
    df = pd.DataFrame([features])
    pred = model.predict(df)
    return int(pred[0])


def find_features_for_animal(animal_name: str, zoo_csv: str = ZOO_CSV) -> Dict[str, int]:
    df = load_data(zoo_csv)
    # match by animal_name case-insensitive
    match = df[df['animal_name'].str.lower() == animal_name.lower()]
    if match.empty:
        return {}
    row = match.iloc[0]
    features = row.drop(labels=['animal_name', 'class_type']).to_dict()
    # ensure ints
    features = {k: int(v) for k, v in features.items()}
    return features


# Flask app
app = Flask(__name__)
label_map = build_label_mapping()


@app.route('/')
def index():
    return jsonify({"status": "ok", "model_exists": os.path.exists(MODEL_PATH)})


@app.route('/train', methods=['POST'])
def train_endpoint():
    result = train_and_save()
    # reload label_map in case class.csv changed
    global label_map
    label_map = build_label_mapping()
    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    payload = request.get_json(force=True)
    model = load_model()
    if model is None:
        return jsonify({"error": "model not found, train first via /train"}), 400

    # Two modes: provide 'animal_name' or provide feature dict
    if 'animal_name' in payload:
        features = find_features_for_animal(payload['animal_name'])
        if not features:
            return jsonify({"error": f"animal '{payload['animal_name']}' not found in zoo.csv"}), 404
    elif 'features' in payload:
        features = payload['features']
    else:
        return jsonify({"error": "provide 'animal_name' or 'features' in JSON body"}), 400

    pred_class = predict_from_features(features, model)
    human_label = label_map.get(pred_class, str(pred_class))
    return jsonify({"predicted_class": int(pred_class), "label": human_label})


def main():
    parser = argparse.ArgumentParser(description='Animal classification API')
    parser.add_argument('--train', action='store_true', help='Train model and save to disk')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    if args.train:
        print('Training model...')
        res = train_and_save()
        print('Done. Accuracy:', res['accuracy'])

    # start Flask app
    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
