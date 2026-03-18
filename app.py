import pandas as pd
import numpy as np
import streamlit as st
import joblib


st.set_page_config(page_title="Cancer Risk Predictor", layout = "centered")

# Load the artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_rf.pkl")
    le = joblib.load("label_encoder.pkl")
    features_names = joblib.load("features_names.pkl")
    return model, le, features_names


model, le, FEATURES_NAMES =load_artifacts()

st.title("Cancer Risk Predictor")
st.markdown("Predict 'Risk Level' (Low/Medium/High) from the patient features.")

# Upload csv or manual input
option = st.radio("Prediction mode", ("Upload CSV (batch)", "Manual Input (single)"))

def preprocess_input(df):
    """ 
    Ensure CSV file has columns in FEATURES_NAMES order and numeric dtype.
    Fills missing columns with 0 and reorders.
    
    """
    #if csv contains extra columns, keep only the features_names columns
    missing = [c for c in FEATURES_NAMES if c not in df.columns]
    if missing:
        st.warning("Missing columns in input - filling {len(missing)} missing columns with zeros: {missing}")
        for c in missing:
            df[c] = 0
    #keep only required columns and in the same order.
    df = df[FEATURES_NAMES].copy()
    #convert to numeric (NAN data)
    df = df.apply(pd.to_numeric, errors = "coerce").fillna(0)
    return df

if option == "Upload CSV (batch)" :
    upload_file = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if upload_file is not None:
        input_df = pd.read_csv(upload_file)
        X = preprocess_input(input_df)
        preds_enc = model.predict(X)
        probs = model.predict_proba(X)
        preds = le.inverse_transform(preds_enc)
        result = X.copy()
        result["Predict_Risk_Level"] = preds
        # attach probability for each class
        for i, cls in enumerate(le.classes_):
            result[f'prob{cls}'] = probs[:, 1]
            st.success("Predictions ready")
            st.dataframe(result)
            st.download_button("Download results (CSV)", result.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

else:
    st.sidebar.header("Patient features (manual)")
    input_data = {}
    # create numeric inputs for each feature (you may want to group or change ranges)
    for feat in FEATURES_NAMES:
        # heuristic default: use 0 as default; change for Age/BMI etc if you prefer
        val = st.sidebar.number_input(feat, value=float(0.0))
        input_data[feat] = val

    if st.sidebar.button("Predict"):
        X_single = pd.DataFrame([input_data])
        X_proc = preprocess_input(X_single)
        pred_enc = model.predict(X_proc)[0]
        probs = model.predict_proba(X_proc)[0]
        pred = le.inverse_transform([pred_enc])[0]

        st.write("### Prediction")
        st.write(f"**Predicted Risk_Level:** {pred}")
        st.write("**Class probabilities:**")
        prob_df = pd.DataFrame({
            'class': list(le.classes_),
            'probability': probs
        }).sort_values('probability', ascending=False).reset_index(drop=True)
        st.table(prob_df)

        # optional: highlight 'High' probability and threshold advice
        high_prob = prob_df.loc[prob_df['class']=='High','probability'].values[0]
        st.info(f"Probability of High risk: {high_prob:.2f}")
        if high_prob >= 0.5:
            st.warning("High risk probability >= 0.5 — consider clinical follow-up.")
        else:
            st.success("High risk probability below 0.5")


# Footer
st.markdown("---")
st.caption("Model: Random Forest(tuned via Optuna).Ensure uploaded CSV has the same feature columns used in training.")
