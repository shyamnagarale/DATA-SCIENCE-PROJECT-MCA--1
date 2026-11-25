import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================
# AI EXPENSE ANALYZER WEB APPLICATION
# =====================================

st.set_page_config(
    page_title="AI Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Load Machine Learning Model
model = joblib.load("expense_model.pkl")

# Title
st.title("💰 AI-Based Personal Expense Pattern & Financial Stress Analyzer")

st.write(
    "Analyze your monthly expenses, predict savings "
    "and understand your financial stress level."
)

st.divider()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Select Option",
    ["Dashboard", "Expense Prediction", "CSV Analysis", "About Project"]
)

# =====================================
# DASHBOARD
# =====================================

if menu == "Dashboard":

    st.header("📊 Expense Dashboard")

    try:
        df = pd.read_csv("analyzed_expense_data.csv")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Income",
            f"₹{df['Income'].sum():,.0f}"
        )

        col2.metric(
            "Total Expenses",
            f"₹{df['Total_Expense'].sum():,.0f}"
        )

        col3.metric(
            "Total Savings",
            f"₹{df['Savings'].sum():,.0f}"
        )

        st.subheader("Expense Data")

        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Average Expense by Category")

        categories = [
            "Food", "Travel", "Shopping",
            "Bills", "Health", "Education"
        ]

        average = df[categories].mean()

        fig, ax = plt.subplots()

        average.plot(kind="bar", ax=ax)

        ax.set_xlabel("Expense Category")
        ax.set_ylabel("Average Amount")
        ax.set_title("Average Monthly Expenses")

        st.pyplot(fig)

    except FileNotFoundError:
        st.error("analyzed_expense_data.csv file not found.")

# =====================================
# EXPENSE PREDICTION
# =====================================

elif menu == "Expense Prediction":

    st.header("🤖 AI Expense Prediction")

    st.write("Enter your monthly income and expenses.")

    income = st.number_input(
        "Monthly Income (₹)", min_value=0
    )

    food = st.number_input(
        "Food Expense (₹)", min_value=0
    )

    travel = st.number_input(
        "Travel Expense (₹)", min_value=0
    )

    shopping = st.number_input(
        "Shopping Expense (₹)", min_value=0
    )

    bills = st.number_input(
        "Bills Expense (₹)", min_value=0
    )

    health = st.number_input(
        "Health Expense (₹)", min_value=0
    )

    education = st.number_input(
        "Education Expense (₹)", min_value=0
    )

    if st.button("Analyze My Expenses"):

        if income <= 0:
            st.error("Please enter income greater than zero.")

        else:

            total_expense = (
                food + travel + shopping +
                bills + health + education
            )

            actual_savings = income - total_expense

            input_data = pd.DataFrame({
                "Income": [income],
                "Food": [food],
                "Travel": [travel],
                "Shopping": [shopping],
                "Bills": [bills],
                "Health": [health],
                "Education": [education]
            })

            prediction = model.predict(input_data)[0]

            ratio = (total_expense / income) * 100

            st.subheader("Expense Analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Expense",
                f"₹{total_expense:,.2f}"
            )

            c2.metric(
                "Actual Savings",
                f"₹{actual_savings:,.2f}"
            )

            c3.metric(
                "Predicted Savings",
                f"₹{prediction:,.2f}"
            )

            st.write(
                f"### Expense Ratio: {ratio:.2f}%"
            )

            if ratio < 50:
                st.success("Low Financial Stress")

            elif ratio < 75:
                st.warning("Medium Financial Stress")

            else:
                st.error("High Financial Stress")

            st.subheader("Expense Breakdown")

            chart_data = pd.DataFrame({
                "Category": [
                    "Food", "Travel", "Shopping",
                    "Bills", "Health", "Education"
                ],
                "Amount": [
                    food, travel, shopping,
                    bills, health, education
                ]
            })

            fig, ax = plt.subplots()

            ax.pie(
                chart_data["Amount"],
                labels=chart_data["Category"],
                autopct="%1.1f%%"
            )

            ax.set_title("Monthly Expense Distribution")

            st.pyplot(fig)

# =====================================
# CSV ANALYSIS
# =====================================

elif menu == "CSV Analysis":

    st.header("📁 Upload Your Expense CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("CSV File Uploaded Successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Dataset Information")

        st.write("Total Records:", len(df))

        st.write("Missing Values:")

        st.write(df.isnull().sum())

        required = [
            "Income", "Food", "Travel", "Shopping",
            "Bills", "Health", "Education"
        ]

        if all(col in df.columns for col in required):

            df = df.dropna(subset=required).copy()

            for col in required:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

            df = df.dropna(subset=required)

            if len(df) > 0:

                df["Total_Expense"] = df[
                    ["Food", "Travel", "Shopping",
                     "Bills", "Health", "Education"]
                ].sum(axis=1)

                df["Savings"] = (
                    df["Income"] - df["Total_Expense"]
                )

                st.subheader("Expense Summary")

                st.write(df.describe())

                st.subheader("Predicted Savings")

                df["Predicted_Savings"] = model.predict(
                    df[required]
                )

                st.dataframe(
                    df.head(20),
                    use_container_width=True
                )

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Analyzed CSV",
                    data=csv,
                    file_name="analyzed_expense_data.csv",
                    mime="text/csv"
                )

            else:
                st.error("CSV मध्ये valid numeric records नाहीत.")

        else:
            st.error(
                "Required columns missing. Please check CSV columns."
            )

# =====================================
# ABOUT PROJECT
# =====================================

elif menu == "About Project":

    st.header("About Our Project")

    st.write("""
    **AI-Based Personal Expense Pattern & Financial Stress Analyzer**

    This project uses Data Science and Machine Learning
    to analyze personal income and expenses.

    Main Features:

    1. Expense Data Analysis
    2. Monthly Expense Visualization
    3. Machine Learning Based Savings Prediction
    4. Financial Stress Analysis
    5. CSV Data Upload
    6. Expense Dashboard
    """)

    st.info(
        "Technology: Python, Pandas, NumPy, "
        "Scikit-learn, Matplotlib and Streamlit."
    )

st.divider()

st.caption("AI Expense Analyzer | Data Science & Machine Learning Project")