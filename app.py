import os
from typing import Dict, Tuple

import pandas as pd
import joblib

# Try to import scikit-learn; show a helpful message in the UI if missing
SKLEARN_AVAILABLE = True
SKLEARN_IMPORT_ERROR = None
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except Exception as e:
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = str(e)

SKLEARN_INSTALL_CMD = "pip install scikit-learn"

BASE_DIR = os.path.dirname(__file__)
ZOO_CSV = os.path.join(BASE_DIR, "zoo.csv")
CLASS_CSV = os.path.join(BASE_DIR, "class.csv")
MODEL_PATH = os.path.join(BASE_DIR, "zoo_rf_model.joblib")


def load_data() -> pd.DataFrame:
    return pd.read_csv(ZOO_CSV)


def build_label_mapping() -> Dict[int, str]:
    if not os.path.exists(CLASS_CSV):
        return {}
    df = pd.read_csv(CLASS_CSV)
    mapping = {}
    if 'Class_Number' in df.columns and 'Class_Type' in df.columns:
        for _, r in df.iterrows():
            mapping[int(r['Class_Number'])] = str(r['Class_Type'])
    else:
        # fallback: try to infer
        for _, r in df.iterrows():
            keys = [c for c in df.columns if 'class' in c.lower() or 'number' in c.lower()]
            vals = [c for c in df.columns if 'type' in c.lower() or 'class' in c.lower()]
            if keys and vals:
                try:
                    mapping[int(r[keys[0]])] = str(r[vals[-1]])
                except Exception:
                    continue
    return mapping


def train_model(df: pd.DataFrame) -> Tuple[object, float]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn missing: {SKLEARN_IMPORT_ERROR}")
    X = df.drop(columns=['animal_name', 'class_type'])
    y = df['class_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    return clf, acc


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def predict_from_features(model, features: Dict[str, int]) -> int:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn missing: {SKLEARN_IMPORT_ERROR}")
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return int(pred)


def get_features_for_animal(df: pd.DataFrame, name: str) -> Dict[str, int]:
    match = df[df['animal_name'].str.lower() == name.lower()]
    if match.empty:
        return {}
    row = match.iloc[0]
    features = row.drop(labels=['animal_name', 'class_type']).to_dict()
    return {k: int(v) for k, v in features.items()}
def main():
    import streamlit as st

    st.set_page_config(page_title='Animal Class Prediction — Manual Input', layout='centered')
    st.title('Animal class prediction — Manual feature entry')

    st.markdown('Enter features manually (no dataset selection). The app will load `zoo_rf_model.joblib` from the project root or you can upload a model file.')

    # show helpful message if sklearn missing
    if not SKLEARN_AVAILABLE:
        st.error('scikit-learn is not installed in this environment.')
        st.info(f'Install with: {SKLEARN_INSTALL_CMD}')
        st.stop()

    # label mapping (optional; used to display human readable label)
    label_map = build_label_mapping()

    # try to load model from disk
    model = None
    try:
        model = load_model()
    except Exception as e:
        st.warning(f'Failed to load existing model: {e}')

    if model is None:
        st.info('No model found at `zoo_rf_model.joblib`. You can upload a .joblib model file below.')
        uploaded = st.file_uploader('Upload model (.joblib)', type=['joblib', 'pkl'])
        if uploaded is not None:
            # save uploaded file to MODEL_PATH and load it
            with open(MODEL_PATH, 'wb') as f:
                f.write(uploaded.getbuffer())
            try:
                model = load_model()
                st.success('Model uploaded and loaded successfully.')
            except Exception as e:
                st.error(f'Failed to load uploaded model: {e}')

    st.markdown('---')
    st.subheader('Enter features')

    # Prepare default values by attempting to read first row of zoo.csv; fallback to zeros
    try:
        df_defaults = load_data()
        sample = df_defaults.drop(columns=['animal_name', 'class_type']).iloc[0]
    except Exception:
        sample = None

    cols = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']

    features = {}
    with st.form('manual_features'):
        for c in cols:
            if c == 'legs':
                default = int(sample[c]) if sample is not None and c in sample.index else 4
                val = st.number_input('legs', min_value=0, max_value=8, value=default)
            else:
                default = int(sample[c]) if sample is not None and c in sample.index else 0
                val = st.selectbox(c, options=[0, 1], index=default)
            features[c] = int(val)

        submitted = st.form_submit_button('Predict')
        if submitted:
            # Use real model prediction when available
            if not SKLEARN_AVAILABLE:
                st.error('scikit-learn is not installed in this environment.')
            elif model is None:
                st.error('No model available. Upload a .joblib model or place `zoo_rf_model.joblib` in the project root.')
            else:
                try:
                    pred = predict_from_features(model, features)
                    label = label_map.get(pred, str(pred))
                    st.success('Prediction complete')
                    st.write(label)
                except Exception as e:
                    st.error(f'Prediction failed: {e}')

    # mapping display removed by user request


if __name__ == '__main__':
    main()
