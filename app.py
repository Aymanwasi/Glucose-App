import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Post-Meal Glucose Change Estimator", page_icon="🩸", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("final_model.joblib")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info


model, info = load_model()
features = info["features"]
ranges = info.get("ranges", {})

SUPPORTED = {
    "Carbs", "Protein", "Fat", "Fiber", "Calories", "Pre_Meal_Glucose",
    "Fasting_GLU", "A1c", "Glucose_Change_30min",
}
unsupported = [f for f in features if f not in SUPPORTED]
if unsupported:
    st.error(f"This app version cannot ask the user for: {unsupported}. Retrain the model without them.")
    st.stop()

st.title("Post-Meal Glucose Change Estimator")
st.caption(
    "Educational demonstration of a machine learning model trained on the CGMacros research dataset "
    "(45 participants). It is not medical advice and must not be used to decide insulin or medication."
)

with st.form("inputs"):
    st.subheader("The meal")
    c1, c2 = st.columns(2)
    carbs = c1.number_input("Carbohydrates (g)", 0.0, 500.0, 45.0, step=1.0)
    protein = c2.number_input("Protein (g)", 0.0, 200.0, 20.0, step=1.0)
    fat = c1.number_input("Fat (g)", 0.0, 200.0, 15.0, step=1.0)
    fiber = c2.number_input("Fibre (g)", 0.0, 100.0, 5.0, step=1.0)
    calc_cal = 4 * carbs + 4 * protein + 9 * fat
    override = st.checkbox("I know the calories (enter them myself)")
    if override:
        calories = st.number_input("Calories (kcal)", 0.0, 3500.0, float(round(calc_cal)), step=10.0)
    else:
        calories = calc_cal
        st.caption(f"Calories estimated from the macronutrients: about {calories:.0f} kcal")

    st.subheader("Your glucose")
    pre = st.number_input("Glucose now, just before the meal (mg/dL)", 40.0, 400.0, 100.0, step=1.0)
    g30 = None
    if "Glucose_Change_30min" in features:
        g30 = st.number_input("Glucose 30 minutes before the meal (mg/dL)", 40.0, 400.0, 100.0, step=1.0,
                               help="Use the reading taken as close to 30 minutes before the meal as possible. The model was trained on FreeStyle Libre sensor readings.")

    fasting = a1c = None
    if "Fasting_GLU" in features or "A1c" in features:
        st.subheader("Your recent lab values")
    if "Fasting_GLU" in features:
        fasting = st.number_input("Fasting glucose (mg/dL)", 50.0, 400.0, 95.0, step=1.0)
    if "A1c" in features:
        a1c = st.number_input("HbA1c (%)", 4.0, 14.0, 5.5, step=0.1)

    go = st.form_submit_button("Estimate")

if go:
    row = {
        "Carbs": carbs, "Protein": protein, "Fat": fat, "Fiber": fiber,
        "Calories": calories, "Pre_Meal_Glucose": pre,
        "Fasting_GLU": fasting, "A1c": a1c,
        "Glucose_Change_30min": None if g30 is None else pre - g30,
    }
    X = pd.DataFrame([row])[features]
    delta = float(model.predict(X)[0])
    est = pre + delta

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Predicted change after 2 hours", f"{delta:+.0f} mg/dL")
    m2.metric("Estimated glucose at 2 hours", f"{est:.0f} mg/dL")

    mae = info.get("test_mae")
    if mae:
        st.info(
            f"On participants the model had never seen, its average error was about {mae:.0f} mg/dL, "
            "so treat this as a rough estimate, not an exact value."
        )

    st.markdown(
        "**General reference only:** about 2 hours after a meal, glucose below roughly 140 mg/dL is typical "
        "for people without diabetes, and below about 180 mg/dL is a common target for adults with diabetes. "
        "Your own targets should come from your healthcare provider."
    )

    outside = []
    for f in features:
        lo_hi = ranges.get(f)
        if lo_hi is not None and X.at[0, f] is not None:
            v = float(X.at[0, f])
            if v < lo_hi[0] or v > lo_hi[1]:
                outside.append(f"{f} = {v:.1f} (training range {lo_hi[0]:.1f} to {lo_hi[1]:.1f})")
    if outside:
        st.warning("Some inputs are outside the range the model was trained on, so the estimate is less reliable: "
                   + "; ".join(outside))

with st.expander("About this model"):
    st.write(
        f"Model: {info.get('model_name', 'Ridge regression')}. Inputs: {', '.join(features)}. "
        f"Held-out test R²: {info.get('test_r2', 'n/a')}, RMSE: {info.get('test_rmse', 'n/a')} mg/dL, "
        f"MAE: {info.get('test_mae', 'n/a')} mg/dL (9 unseen participants)."
    )
    st.write(
        "Target: change in glucose from just before a meal to about 2 hours after it. "
        "Training data: CGMacros (45 participants, free-living, continuous glucose monitoring)."
    )