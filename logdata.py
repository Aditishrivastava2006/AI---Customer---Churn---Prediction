import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Churn Predictor", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:35px !important;
    color:#4CAF50;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🤖 AI Customer Churn Predictor</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Information")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    if len(numeric_cols) > 0:
        col = st.selectbox("Select Feature", numeric_cols)

        fig = px.histogram(df, x=col, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train AI Model"):
        X = df.drop(target, axis=1)
        y = df[target]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.success(f"🎯 Model Accuracy: {acc:.2f}")

        prediction_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

        st.dataframe(prediction_df)

        csv = prediction_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )