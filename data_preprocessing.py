import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle

enable_plot = False


def shorten_categories(categories, cutoff):
    """
    :param categories:
    :param cutoff:
    :return: categorical_map dictionary
    """
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map


def clean_work_xp(xp_num):
    if xp_num == "More than 50 years":
        return 50
    if xp_num == "Less than 1 year":
        return 0.5
    return float(xp_num)


def clean_education(level):
    if level.__contains__("Bachelor’s degree"):
        return "Bachelor's Degree"
    if level.__contains__("Master's degree"):
        return "Master's Degree"
    if level.__contains__("Professional degree"):
        return "Post Grad"
    if level.__contains__("Other doctoral"):
        return "Post Grad"
    return "Less than a Bachelors"


url = "data/survey_results_public.csv"
data_salaries = pd.read_csv(url, encoding="utf-8", header=0)
# print(data_salaries.head(5))
# print(data_salaries.columns)
# print(data_salaries.shape)

# take only the 5 columns relevant to this analysis
data_salaries = data_salaries[
    ["Country", "EdLevel",
     "YearsCodePro",
     "Employment",
     "ConvertedCompYearly"]]
# rename this column to salaries
data_salaries.rename({
    "ConvertedCompYearly": "Salary"
}, axis=1, inplace=True)
# drop all null values
data_salaries = data_salaries.dropna()
# print(data_salaries.isnull().sum())

data_salaries = data_salaries[data_salaries["Employment"] == "Employed, full-time"]
# drop Employment column since we do not need this for prediction
# or exploration
data_salaries.drop("Employment",
                   axis=1,
                   inplace=True)

# print(data_salaries["Country"].value_counts())
country_map = shorten_categories(data_salaries["Country"].value_counts(), 400)
data_salaries["Country"] = data_salaries["Country"].map(country_map)

# print(data_salaries["Country"].value_counts())
# print(data_salaries.head(10))

# commented out since this part of the preprocessing
# has been completed
if enable_plot:
    fig, ax = plt.subplots(1, 1,
                           figsize=(12, 7))
    data_salaries.boxplot("Salary", "Country", ax=ax)
    plt.suptitle("Salary ($USD) vs Country")
    plt.title("")
    plt.ylabel("Salary ($USD)")
    plt.xticks(rotation=90)
    plt.show()

data_salaries = data_salaries[data_salaries["Salary"] <= 200000]
data_salaries = data_salaries[data_salaries["Salary"] > 10000]
data_salaries = data_salaries[data_salaries["Country"] != "Other"]

# print(data_salaries["YearsCodePro"].unique())

# clean up years of professional coding experience column
data_salaries["YearsCodePro"] = data_salaries["YearsCodePro"].apply(clean_work_xp)
# clean up education column
# data_salaries["EdLevel"] = data_salaries["EdLevel"].apply(clean_education)
# print(data_salaries["EdLevel"].unique())
data_salaries["EdLevel"] = data_salaries["EdLevel"] \
    .replace(['Bachelor’s degree (B.A., B.S., B.Eng., etc.)'],
             "Bachelor's Degree")
data_salaries["EdLevel"] = data_salaries["EdLevel"] \
    .replace(['Master’s degree (M.A., M.S., M.Eng., MBA, etc.)'],
             "Master's Degree")
data_salaries["EdLevel"] = data_salaries["EdLevel"] \
    .replace(['Professional degree (JD, MD, Ph.D, Ed.D, etc.)'],
             "Post Grad")
data_salaries["EdLevel"] = data_salaries["EdLevel"] \
    .replace(['Some college/university study without earning a degree',
              'Associate degree (A.A., A.S., etc.)',
              'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
              'Primary/elementary school',
              'Something else'],
             "Less than a Bachelors'")
# print(data_salaries["EdLevel"].unique())

# print(data_salaries)
# output a pickle file
# this is for the exploration part of the web app
data_salaries.to_pickle("data/data_explore.pkl")

# Further Processing
LabelEncoder_education = LabelEncoder()
data_salaries["EdLevel"] = LabelEncoder_education.fit_transform(data_salaries["EdLevel"])
LabelEncoder_country = LabelEncoder()
data_salaries["Country"] = LabelEncoder_country.fit_transform(data_salaries["Country"])

# print(data_salaries["EdLevel"].unique())
# print(data_salaries)
data_salaries.to_pickle("data/data_predict.pkl")
# take the fitted encoders as pickles
pickle.dump(LabelEncoder_education,
            open("pickles/le_edlvl.pkl", "wb"))
pickle.dump(LabelEncoder_country,
            open("pickles/le_country.pkl", "wb"))
