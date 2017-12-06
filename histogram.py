import pandas as pd

census80 = pd.read_csv("Data/census80.csv")
census80i = census80[["educ"]]
census80i.to_csv("Data/census80i.csv")