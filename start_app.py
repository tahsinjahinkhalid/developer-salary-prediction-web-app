# import relevant modules
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
# for data visualisation
import plotly.express as px
# for machine learning
from sklearn.ensemble import RandomForestRegressor
import pickle

# configure the streamlit page
st.set_page_config(layout="centered",
                   page_title="Software Developer Salaries",
                   page_icon=":alembic:")

st.markdown("""
# :computer: Software Developer Salary
## :star: Prediction/Exploration Web App :star:
## By: [Tahsin Jahin Khalid](https://tahsinjahinkhalid.github.io)
""")

st.markdown("""
## :bulb: About this Project
- [Dataset Source Link](https://insights.stackoverflow.com/survey/)
- Data from the Stack Overflow 2023 Developer Salary Survey has been used to develop a web app using Streamlit for predicting the salary based on selected parameters and also exploration of the dataset.
""")

# configure menu
selected_tab = option_menu(
    menu_title="App Menu",
    options=["Prediction", "Exploration"],
    icons=["bar-chart-line", "clipboard2-pulse"],
    orientation="horizontal",
    menu_icon="cast",
    default_index=1
)

if selected_tab == "Prediction":
    pickles = ["pickles/model_rf.pkl",
               "pickles/le_country.pkl",
               "pickles/le_edlvl.pkl"]

    # load the save content
    # into new variables here
    with open(pickles[0], "rb") as file:
        rf_mdl = pickle.load(file)
    with open(pickles[1], "rb") as file:
        le_country = pickle.load(file)
    with open(pickles[2], "rb") as file:
        le_edlvl = pickle.load(file)
    # st.write("Prediction Tab")

    st.markdown("## Software Developer Salary Prediction")

    countries = [
        'United States of America',
        'United Kingdom of Great Britain and Northern Ireland',
        'Australia',
        'Netherlands',
        'Germany',
        'Sweden',
        'France',
        'Spain',
        'Brazil',
        'Italy',
        'Canada',
        'Switzerland',
        'India',
        'Norway',
        'Denmark',
        'Israel',
        'Poland']

    education = (
        "Less than a Bachelors'",
        "Bachelor's Degree",
        "Master's Degree",
        "Post Grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    confirm = st.button("Estimated Salary")
    if confirm:
        test_case = np.array([[country, education, experience]])
        test_case[:, 0] = le_country.transform(test_case[:, 0])
        test_case[:, 1] = le_edlvl.transform(test_case[:, 1])
        test_case = test_case.astype(float)

        salary = rf_mdl.predict(test_case)
        st.subheader(f"Estimated Salary: ${salary[0]:.2f}")

if selected_tab == "Exploration":
    st.markdown("## Software Developer Salary Exploration")

    with open("data/data_explore.pkl", "rb") as file:
        data_salaries = pickle.load(file)

    # st.dataframe(data_salaries)
    # print(data_salaries["EdLevel"].unique())
    # bar_graph = px.bar(data_salaries, "EdLevel", "Salary")
    # st.plotly_chart(bar_graph)\

    # st.write(data_salaries["EdLevel"].unique())

    data_country = data_salaries["Country"].value_counts()

    # Viz 1: pie Chart
    st.subheader("Survey Participants by Country")
    # st.dataframe(data_country)
    pie_viz = px.pie(data_frame=data_country,
                     names=data_country.index,
                     labels="Country",
                     values="count")
    pie_viz.update_layout()
    st.plotly_chart(pie_viz)

    # Viz 2: median salary by country
    st.subheader("Median Salary based on Country")
    dataf2 = data_salaries \
        .groupby(["Country"])["Salary"] \
        .median() \
        .sort_values(ascending=True)

    dataf2 = dataf2.reset_index()
    dataf2["Country"] = dataf2["Country"].replace(["United States of America"], "USA")
    dataf2["Country"] = dataf2["Country"].replace(["United Kingdom of Great Britain and Northern Ireland"], "UK")

    # st.dataframe(dataf2)

    bar_viz = px.bar(dataf2, x="Country", y="Salary")
    bar_viz.update_layout(
        xaxis_title="Country",
        yaxis_title="Median Salary",
        font={"family": "monospace"}
    )
    st.plotly_chart(bar_viz)

    # Viz 3: median salary based on experience
    st.subheader("Median Salary based on Professional Experience")
    dataf3 = data_salaries.groupby(["YearsCodePro"])["Salary"].median().sort_values(ascending=True)

    scatter_viz = px.scatter(dataf3,
                             x=dataf3.index,
                             y="Salary",
                             opacity=0.65,
                             trendline='ols',
                             trendline_color_override='darkblue')
    scatter_viz.update_layout(
        xaxis_title="Years of Professional Experience",
        yaxis_title="Median Salary",
        font={
            "family": "Courier New, monospace",
            "size": 18,
            "color": "RebeccaPurple"}
    )
    st.plotly_chart(scatter_viz)
