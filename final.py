"""
Name: Nathan Carrillo
CS230: section 1
Data: new_england_airports
URL: Link

Description:
This program will be able to find the oldest record from the dataset, the most recent, the weapon deployment location
"""
from pickletools import long1

import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

#read in data
def read_data():
    return pd.read_csv("new_england_airports.csv").set_index("id")

#Title
st.title("New England Airport Data Explorer")

#filter the data
df = read_data()
df = df.loc[df['iso_region'].isin(["US-CT", "US-MA","US-NH","US-RI","US-VT","US-ME"])]


#[DA1] Clean the data:
clean_df = df.dropna(subset=["latitude_deg","longitude_deg","iso_region","type", "name"])

#[DA7] Create lists of states and airports types
states = sorted(clean_df["iso_region"].unique())
airport_types = sorted(clean_df["type"].unique())

#[ST1] Dropdown for state selection
selected_state = st.sidebar.selectbox("Select a U.S. State(Region):", states)

#[ST2] Multiselect for airport types
selected_types = st.sidebar.multiselect("Select Airport Types:", airport_types, default=airport_types)


#[PY1] Calculates the distances between airports; Query 3 to show all airports within x distance of airport
from math import radians, sin, cos, sqrt, atan2
def haversine(lat1,lon1,lat2,lon2):
    R = 3437.75 #Earth's radius in nautical miles
    dlat = radians(lat2-lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a),sqrt(1-a))
    return R * c

#[DA4][DA5] Filter the clean_df based on Streamlit inputs
filtered_df = clean_df[
    (clean_df["iso_region"] == selected_state) &
    (clean_df["type"].isin(selected_types))
]

#[DA9] Create a lebel for airport dropdown
filtered_df["label"] = filtered_df["name"] + " (" + filtered_df["municipality"].fillna("Unknown") + ")"

#[ST3][ST4] Sidebar controls, slider
st.sidebar.title("Airport Distance Explorer")
selected_airport = st.sidebar.selectbox("Choose an airport:", filtered_df["label"])
max_distance_nm = st.sidebar.slider("Distance (nautical miles):",10,250,50)

#get coordinates of selected airport
selected_row = filtered_df[filtered_df["label"] == selected_airport].iloc[0]
lat1,lon1 = selected_row["latitude_deg"], selected_row["longitude_deg"]

#[DA9] Add a distance column
filtered_df["distance_nm"] = filtered_df.apply(
    lambda row: haversine(lat1,lon1, row["latitude_deg"], row["longitude_deg"]),
    axis = 1
)

#[DA4] Filter airports within selected radius
nearby_df = filtered_df[filtered_df["distance_nm"] <= max_distance_nm]

#Header
st.subheader("New England Airport Map")

#[MAP] PyDeck Map showing selected and nearby airports
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude = lat1,
        longitude = lon1,
        zoom= 6,
        pitch= 50,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=nearby_df,
            get_position= '[longitude_deg, latitude_deg]',
            get_radius= 10000,
            get_fill_color='[200,300,0,160]',
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([selected_row]),
            get_position='[longitude_deg, latitude_deg]',
            get_radius= 15000,
            get_fill_color='[0,0,200,200]',
            pickable=True,
        )
    ],
    tooltip={"text":"{name} ({municipality})\n{distance_nm:2f} nm"}
))

#[Chart1] Show a table of nearby airports
st.write("Airports within", max_distance_nm, "nautical miles of", selected_airport)
st.dataframe(nearby_df[["name","municipality","distance_nm"]].sort_values("distance_nm"))

#[DA4]
clean_df = df.dropna(subset=["iso_region","scheduled_service","municipality","name"])
states = sorted(clean_df["iso_region"].unique())
#[ST3] Dropdown for state selection
selected_state = st.sidebar.selectbox("Select a U.S.(Region) for Scheduled Service Info:", states)

#[DA4] Filter by selected state
states_df = clean_df[clean_df["iso_region"]== selected_state]

#[Chart2] Pie chart for scheduled service
st.subheader(f"Scheduled Service in {selected_state}")
service_counts = states_df["scheduled_service"].value_counts()

#[PY4] List comprehension used for labels
labels = [f"{val} ({count})" for val, count in service_counts.items()]
colors = ["green","red"] if "yes" in service_counts.index else ["red","green"]

fig,ax = plt.subplots()
ax.pie(service_counts, labels=labels, autopct="%1.1f%%",startangle=90,colors=colors)
ax.axis("equal")
plt.title("Proportion of Airports with Scheduled Commercial Service")
st.pyplot(fig)

#[Chart3] Show the relevant airport info in a table
st.subheader(f"Airports in {selected_state}")
st.dataframe(states_df[["name","municipality","scheduled_service"]].reset_index(drop=True))

#[ST1] Dropdown for state
selected_state = st.sidebar.selectbox("Select a state to view airport types:", sorted(clean_df["iso_region"].unique()))

#[DA4] Filter by selected state
states_df = clean_df[clean_df["iso_region"] == selected_state]

#[PY2] Function that returns dictionary and sorted keys
def get_type_distribution(df):
    type_counts = df["type"].value_counts().to_dict()
    sorted_types = sorted(type_counts.keys())
    return type_counts, sorted_types
#Call the function
type_counts_dict, type_order = get_type_distribution(states_df)

#[PV5] Use of dictionary keys and values
counts = [type_counts_dict[key] for key in type_order]

#[Chart4] Bar chart using plt
fig,ax = plt.subplots()
ax.bar(type_order,counts,color='yellow')
ax.set_title(f"Airport Types in {selected_state}")
ax.set_xlabel("Airport Type")
ax.set_ylabel("Number of Airports")
plt.xticks(rotation=45)

st.pyplot(fig)