# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/try_eda/genz_college_admission_prediction.csv"
)
df.sample(5)
# %%
df.info()

# %%
df.isnull().sum()
# %%
df.describe()
# %%
df.duplicated().sum()

# %%
df.shape

# %%
df.corr()
# so making data less so i can do eda my laptop crashed twice lol
# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1, random_state=42)

# %%


# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/try_eda/test.csv")

# %%
df.shape
# %%
df.info()
# %%
df.describe()
# %%
df.isnull().sum()
# %%
df.duplicated().sum()
# %%
# univariate analysis
#  categorical column
sns.countplot(x=df["admission_status"])

# %%
df["admission_status"].value_counts().plot(kind="pie", autopct="%.2f")

# %%
df["gender"].value_counts().plot(kind="pie", autopct="%.2f")

# %%
sns.countplot(x=df["leadership_positions"])

# %%
df["leadership_positions"].value_counts().plot(kind="pie", autopct="%.2f")
# %%
# numerical columns
sns.histplot(x=df["age"], kde=True)
df["age"].skew()
# %%
sns.displot(x=df["age"])
# %%
sns.boxplot(x=df["age"])
# %%
sns.histplot(x=df["state"])
# %%
sns.countplot(x=df["state"])
# %%
sns.displot(x=df["family_income"], kde=True)
df["family_income"].skew()
# %%
sns.boxplot(x=df["family_income"])
# %%
sns.displot(x=df["high_school_gpa"], kde=True)
df["high_school_gpa"].skew()
# %%
sns.displot(x=df["sat_score"])
df["sat_score"].skew()
# %%
sns.displot(x=df["act_score"], kde=True)
df["act_score"].mean()
df["act_score"].skew()
# %%
sns.displot(x=df["attendance_rate"], kde=True)
# %%
sns.displot(x=df["ap_courses"])
# %%
sns.boxplot(x=df["ap_courses"])
# %%
sns.displot(x=df["extracurricular_count"], kde=True)
df["extracurricular_count"].skew()
# %%
sns.boxplot(x=df["extracurricular_count"])
# %%
sns.displot(x=df["volunteer_hours"], kde=True)
# %%
sns.boxplot(x=df["volunteer_hours"])
# %%
sns.displot(x=df["coding_projects"])
# %%
sns.boxplot(x=df["coding_projects"])
# %%
sns.displot(x=df["social_media_hours"], kde=True)
# %%
sns.boxplot(x=df["social_media_hours"])
# %%
sns.displot(x=df["online_certifications"], kde=True)
# %%
sns.countplot(x=df["online_certifications"])
# %%
sns.displot(x=df["essay_score"], kde=True)
# %%
sns.histplot(x=df["essay_score"], kde=True)
# %%
sns.boxplot(x=df["essay_score"])
# %%
sns.displot(x=df["recommendation_score"])
# %%
sns.boxplot(x=df["recommendation_score"])
# %%
sns.boxplot(x=df["interview_score"])
# %%
sns.histplot(x=df["interview_score"], kde=True)
# %%
# multivarite analysis
sns.scatterplot(x=df["family_income"], y=df["age"], size=df["high_school_gpa"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["age"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["sat_score"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["act_score"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["volunteer_hours"])
# %%
sns.scatterplot(x=df["sat_score"], y=df["act_score"])
# %%
sns.scatterplot(x=df["sat_score"], y=df["volunteer_hours"], hue=df["admission_status"])
# %%
sns.scatterplot(
    x=df["sat_score"], y=df["social_media_hours"], style=df["admission_status"]
)
# %%
sns.scatterplot(x=df["sat_score"], y=df["interview_score"])
# %%
sns.scatterplot(
    x=df["sat_score"],
    y=df["act_score"],
    size=df["coding_projects"],
    hue=df["admission_status"],
)
# %%
sns.displot(x=df["age"], y=df["admission_status"])
# %%
sns.pairplot(df)
# %%
from pandas_profiling import ProfileReport

prof = ProfileReport(df)
prof.to_file(output_file="output.html")
# %%
# from now i did corealtion of data
sns.barplot(x=df["admission_status"], y=df["ap_courses"])
# %%
sns.displot(x=df[df["admission_status"] == 0]["ap_courses"], kde=True)
sns.displot(x=df[df["admission_status"] == 1]["ap_courses"], kde=True)
# %%
sns.displot(x=df["ap_courses"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["high_school_gpa"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.barplot(x=df["admission_status"], y=df["high_school_gpa"])
# %%
sns.barplot(x=df["admission_status"], y=df["sat_score"])
# %%
sns.displot(x=df["sat_score"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(
    x=df["extracurricular_count"], hue=df["admission_status"], kind="kde", fill=True
)
# %%
sns.heatmap(pd.crosstab(df["act_score"], df["extracurricular_count"]))
# %%
sns.heatmap(pd.crosstab(df["act_score"], df["gender"]))
# %%
sns.displot(x=df["act_score"], hue=df["gender"], kind="kde", fill=True)
# %%
sns.displot(x=df["interview_score"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.heatmap(pd.crosstab(df["state"], df["act_score"]))
# from here will analysis of admission_status cloumn
# %%
sns.displot(x=df["age"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["family_income"], hue=df["admission_status"], kind="kde", fill=True)

# %%
sns.displot(x=df["high_school_gpa"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["sat_score"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["act_score"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["attendance_rate"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["attendance_rate"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(x=df["ap_courses"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(
    x=df["extracurricular_count"], hue=df["admission_status"], kind="kde", fill=True
)
# %%
sns.displot(x=df["volunteer_hours"], hue=df["admission_status"], kind="kde", fill=True)

# %%
sns.displot(
    x=df["leadership_positions"], hue=df["admission_status"], kind="kde", fill=True
)
# %%
sns.displot(x=df["coding_projects"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(
    x=df["social_media_hours"], hue=df["admission_status"], kind="kde", fill=True
)
# %%
sns.displot(
    x=df["online_certifications"], hue=df["admission_status"], kind="kde", fill=True
)
# %%
sns.displot(x=df["essay_score"], hue=df["admission_status"], kind="kde", fill=True)
# %%
sns.displot(
    x=df["recommendation_score"], hue=df["admission_status"], kind="kde", fill=True
)
# %%
sns.displot(x=df["interview_score"], hue=df["admission_status"], kind="kde", fill=True)
# %%
df.groupby("social_media_hours")["admission_status"].mean() * 100
# %%
df.groupby("ap_courses")["admission_status"].mean() * 100
# %%
df.groupby("ap_courses")["admission_status"].mean() * 100
# %%
"""
# ye 600 me sa hai

gender - in whole data gender iss equal (male 49.03% = female 49.94 %) and rest other (2.02 %)

age -all are of same age

state- state in total 10 nad all state have equal number of ppl

high school gpa - 2.5 is min gpa for the ppls
max ppl have gpa in range 2.5 to 3.5
max ppl have 4 gpa
range of gpa is 0 to 4

family income - max ppl lie in 0 to 100000
many outliers

sat score-1100 and 1600 max ppl lie in this region
0- 1600 is the range ig

act score- 25 marks are obtained as highest accuring
0 to 35 range

attendance rate - most ppl have 100% attendance

ap courses- max ppl have done 2 -4 couses

extra curricular count- max ppl in 2 to 5 range
have outliers

volunteer hours - 0 - 200 to max ppl aare there
have a lot outliers

coding projects - many ppl have 1 or 2 projects

socisal media hours - many ppl use  to spend their 2 to 6 hours on it
many outliers

onlie certificate - many ppl have 1 or 2 certificates

essay score - maxx ppl have 100

recommendation score - max ppl habvve scored 80 and 100

intervirew score - max ppl have scored in range of 70 -80

admission status- 88.2 % =1 or 11.80 ==0

# MUltivarriate
ppl having high school gpa greater than 3 gpa have high possibility of getting admission
ppl who have ap couses in range of 2 -4 have higer admission selection or might be possible like mostly ppl have done 2 4 couses so it dont matter and if ap couses greater tha 6 more higher chanses
ppl who score highere than 1400 ae getting admission and most ppl are getting admission ther sat score is is in range 800 to 1200
most ppl have done extra curricular in range of 2 to 6 and got admitted
and on the basis of act score both gender have almost same score
interview score most got admitted in range of 60 to 90
all state have alsmosst same act score
age doont matter every age got same selection citeria
Admission probability increases continuously with GPA. No special importance of 2.7.
ppl having score greater than 1000 are getting admission
ppl with 100 % attendance rate are getting admission
screen time dosent matter

"""
