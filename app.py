import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

df = pd.read_csv("Project_8\open_pubs.csv")

# Drop rows where 'Latitude' is '\N'
df = df[df['Latitude'] != r'\N']

# Drop rows where 'Longitude' is '\N'
df = df[df['Longitude'] != r'\N']

# Convert 'Latitude' and 'Longitude' columns to float data type
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# Extract the 'Latitude' and 'Longitude' columns from the DataFrame
latitude = df['Latitude'].values
longitude = df['Longitude'].values

# Combine the 'Latitude' and 'Longitude' arrays into a single 2D array
lat_lon_array = np.column_stack((latitude, longitude))


# Create Clusters. Assume with no. of clusters = 2
kmeans = KMeans(n_clusters=5)  
kmeans.fit(lat_lon_array) 

# Create a scatter plot with cluster labels and centroids
fig, ax = plt.subplots()
scatter = ax.scatter(lat_lon_array[:, 0], lat_lon_array[:, 1], c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
ax.set_title('K-means Clustering')  # Replace with your plot title
ax.grid(True)
plt.colorbar(scatter, ax=ax)

st.set_page_config(page_title="Pub Finder Application", page_icon="🍺")

# Set the default page as 'Homepage' in the sidebar
default_page = st.sidebar.radio('Thank you for choosing the PUB FINDER APPLICATION!', options=['Homepage', 'Pub Location', 'Find Nearest Pub'], index=0)

# Check the selected page and show corresponding content
if default_page == 'Homepage':
    # Add homepage content here
    st.title('Welcome to the Pub Finder App!')
    st.write("Our pub finder app uses your postcode and Local Authority to locate nearby pubs or bars. With input of latitude and longitude values, you can easily find the best pub for your preferences. Cheers to a seamless pub-hopping experience with our app!")
    st.image("https://www.churchillarmskensington.co.uk/-/media/sites/pubs-and-hotels/c/the-churchill-arms-kensington-_-p021/images/gallery-2022/churchill-gallery-8.ashx?h=1280&w=1920&la=en&hash=ED6C30F0AD92939FF62255B2C5BF56F7")
    st.markdown("The following is the usage of K-means nearest neighbors and finding the centroids:")
    st.pyplot(fig)
elif default_page == 'Pub Location':
    # Add pub location content here
    st.title("View Pub Locations on the Map")
    option = st.selectbox(
    'Choose the option to see the location',
    ('Postal Code', 'Local Authority'))
    if option == 'Postal Code':
        postal_code = st.selectbox('Enter Postal Code', df['Postcode'].unique())
        filtered_df = df[df['Postcode'] == postal_code]
        latitude = filtered_df['Latitude'].mean()
        longitude = filtered_df['Longitude'].mean()
        m = folium.Map(location=[latitude, longitude], zoom_start=14)
        for index, row in filtered_df.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Name_Of_The_Restaurant']).add_to(m)
        folium_static(m)  # Render the map using st_folium
    else:
        local_authority = st.selectbox("Enter Local Authority", df["Local_authority"].unique())
        fill_df = df[df['Local_authority'] == local_authority]
        fill_df['Latitude'] = pd.to_numeric(fill_df['Latitude'], errors='coerce')
        fill_df['Longitude'] = pd.to_numeric(fill_df['Longitude'], errors='coerce')
        latitude = fill_df['Latitude'].mean()
        longitude = fill_df['Longitude'].mean()
        map = folium.Map(location=[latitude, longitude], zoom_start=14)
        for index, row in fill_df.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Name_Of_The_Restaurant']).add_to(map)
        folium_static(map)

else:
    # Add find nearest pub content here
    st.title("Find the Nearest Pub based on your Location")
    user_lat = st.number_input("Enter Latitude of your region")
    user_lon = st.number_input("Enter Longitude of your region")

    # Convert Latitude and Longitude columns to numeric data type
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Calculate Euclidean distance between user's location and each pub location
    df['Distance'] = df.apply(lambda row: euclidean((user_lat, user_lon), (row['Latitude'], row['Longitude'])), axis=1)

    # Sort pubs by distance and select the nearest 5 pubs
    nearest_pubs = df.sort_values(by='Distance').iloc[:5]

    # Create a folium map object
    pub_location = folium.Map(location=[user_lat, user_lon], zoom_start=14)

    for index, row in nearest_pubs.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        pub_name = row['Name_Of_The_Restaurant']
        folium.Marker([lat, lon], popup=pub_name).add_to(pub_location)

    folium_static(pub_location)
