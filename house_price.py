import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error ,mean_squared_error
import numpy as np

real_dataset = pd.read_csv("messy_india_house_prices_dataset.csv")
df = real_dataset.copy()
df.dropna(inplace=True)

df["property_type"] = df["property_type"].replace({"Villa":"villa"})
df = pd.get_dummies(df,columns=["locality"])
df = pd.get_dummies(df,columns=["property_type"])
df = pd.get_dummies(df,columns=["furnishing_status"])

'''df["locality_Andheri"] = df["locality_Andheri"].astype(int)
df["locality_Banjara Hills"] = df["locality_Banjara Hills"].astype(int)
df["locality_Dwarka"] = df["locality_Dwarka"].astype(int)
df["locality_Gomti Nagar"] = df["locality_Gomti Nagar"].astype(int)
df["locality_Hinjewadi"] = df["locality_Hinjewadi"].astype(int)
df["locality_Malviya Nagar"] = df["locality_Malviya Nagar"].astype(int)
df["locality_Navrangpura"] = df["locality_Navrangpura"].astype(int)
df["locality_Salt Lake"] = df["locality_Salt Lake"].astype(int)
df["locality_T Nagar"] = df["locality_T Nagar"].astype(int)
df["locality_Whitefield"] = df["locality_Whitefield"].astype(int)
df["property_type_Apartment"] = df["property_type_Apartment"].astype(int)
df["property_type_Flat"] = df["property_type_Flat"].astype(int)
df["property_type_Independent House"] = df["property_type_Independent House"].astype(int)
df["property_type_apartment"] = df["property_type_apartment"].astype(int)
df["property_type_villa"] = df["property_type_villa"].astype(int)
df["furnishing_status_Furnished"] = df["furnishing_status_Furnished"].astype(int)
df["furnishing_status_Semi-Furnished"] = df["furnishing_status_Semi-Furnished"].astype(int)
df["furnishing_status_Unfurnished"] = df["furnishing_status_Unfurnished"].astype(int)
df["furnishing_status_semi furnished"] = df["furnishing_status_semi furnished"].astype(int)'''

convert_label = ["locality_Andheri",
                 "locality_Banjara Hills",
                 "locality_Dwarka",
                 "locality_Gomti Nagar",
                 "locality_Hinjewadi",
                 "locality_Malviya Nagar",
                 "locality_Navrangpura",
                 "locality_Salt Lake",
                 "locality_T Nagar",
                 "locality_Whitefield",
                 "property_type_Apartment",
                 "property_type_Flat",
                 "property_type_Independent House",
                 "property_type_apartment",
                 "property_type_villa",
                 "furnishing_status_Furnished",
                 "furnishing_status_Semi-Furnished",
                 "furnishing_status_Unfurnished",
                 "furnishing_status_semi furnished"]

for i in range(len(convert_label)):
    df[convert_label[i]] = df[convert_label[i]].astype(int)

df["floor"] = df["floor"].replace({"Top":100,
                                   "Ground":0,})

X = df[["bedrooms","bathrooms","area_sqft","age_years","parking_spaces","floor"]]

scaler = StandardScaler()
df[X.columns] = scaler.fit_transform(X)


X_scaled = df[["bedrooms","bathrooms",
               "area_sqft",
               "age_years",
               "parking_spaces",
               "floor",
               "locality_Andheri",
               "locality_Banjara Hills",
               "locality_Dwarka",
               "locality_Gomti Nagar",
               "locality_Hinjewadi",
               "locality_Malviya Nagar",
               "locality_Navrangpura",
               "locality_Salt Lake",
               "locality_T Nagar",
               "locality_Whitefield",
               "property_type_Apartment",
               "property_type_Flat",
               "property_type_Independent House",
               "property_type_apartment",
               "property_type_villa",
               "furnishing_status_Furnished",
               "furnishing_status_Semi-Furnished",
               "furnishing_status_Unfurnished",
               "furnishing_status_semi furnished"]]

y = df[["price_inr"]]
X_train, X_test, y_train, y_test = train_test_split(X ,y ,random_state=42 ,test_size=0.2)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print(round(mae))
print(round(mse))
print(round(rmse))






