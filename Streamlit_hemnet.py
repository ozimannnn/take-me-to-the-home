import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from folium import plugins
from PIL import Image

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import folium
from folium.plugins import HeatMap, MousePosition
import random
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    mean_squared_log_error, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from streamlit_folium import st_folium
import base64
from streamlit_folium import folium_static
import plotly.express as px
import altair as alt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css('hemnet.css')
pd.set_option('display.max_columns', 500)


@st.cache_data
def load_data():
    df_ = pd.read_csv('hemnet_last.csv')
    df_['sold_date'] = pd.to_datetime(df_['sold_date'], format='%Y-%m-%d')
    df_['build_year'] = pd.to_datetime(df_['build_year'], format='%Y-%m-%d')
    return df_.copy()


df = load_data()


@st.cache_data
def load_other_data():
    population_df = pd.read_csv("sweden_population.csv")
    education_df = pd.read_csv("sweden_education.csv")
    df['property_type'] = df['property_type'].astype(str)
    transport_df = pd.read_excel(
        'ilçe ve yıllara göre toplu taşıma seferleri.xlsx', header=4)
    transport_df.columns = ['County', 'Type', '2019', '2020', '2021', '2022']
    transport_df = transport_df.drop(index=0)
    transport_df['County'] = transport_df['County'].str.replace(' län', '')
    # Türleri İngilizce'ye çevirme
    type_translation = {
        'Buss': 'Bus',
        'Tåg': 'Train',
        'Spårvagn': 'Tram',
        'Tunnelbana': 'Subway',
        'Båt': 'Boat',
        'Regional linjetrafik': 'Regional Line Traffic',
        'RVU': 'RVU'
    }
    transport_df['Type'] = transport_df['Type'].map(type_translation)
    return population_df, education_df, transport_df


population_df, education_df, transport_df = load_other_data()


def grab_col_names(dataframe, cat_th=16, car_th=25):
    """
    Grab column names based on their types and cardinality.

    Parameters:
    cat_th (int): Threshold for numerical columns to be considered categorical. Default is 10.
    car_th (int): Threshold for categorical columns to be considered cardinal. Default is 20.

    Returns:
    tuple: Lists of categorical columns, numerical columns, and cardinal columns.
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['int', 'float', 'int64', 'float64']]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ['category', 'object']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int', 'float', 'int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]
    date_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'datetime64[ns]']

    return cat_cols, num_cols, cat_but_car, date_cols


cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(df)


def dataframe_summary(dataframe: pd.DataFrame, response_var: str, head: int = 5, q1: float = 0.25, q2: float = 0.50,
                      q3: float = 0.75, q4: float = 0.95, verbose: bool = False):
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    def print_section(title):
        if verbose:
            print(f"\n{'-' * 20} {title} {'-' * 20}\n")

    cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(dataframe)
    memory_usage = dataframe.memory_usage(deep=True)

    summary_df = pd.DataFrame({
        'Data Type': dataframe.dtypes,
        'Unique Values': dataframe.nunique(),
        'Memory Usage (MB)': memory_usage / 1e6,
        'Null Value Ratio (%)': dataframe.isnull().mean() * 100,
        'Zero Value Ratio (%)': (dataframe == 0).mean() * 100,
        'Skewness': dataframe.select_dtypes(include='number').skew(),
        'Kurtosis': dataframe.select_dtypes(include='number').kurt()
    }).round({
        'Memory Usage (MB)': 3,
        'Null Value Ratio (%)': 3,
        'Zero Value Ratio (%)': 3,
        'Skewness': 3,
        'Kurtosis': 3
    }).reset_index().rename(columns={'index': 'Variable'})
    summary_df['Column Type'] = summary_df.Variable.map(
        lambda col: 'Categorical' if col in cat_cols else
        'Numerical' if col in num_cols else
        'Cardinal Categorical' if col in cat_but_car else
        'Date' if col in date_cols else 'Other'
    )

    summary_df['Correlation with Response'] = np.nan

    if response_var in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[response_var]):
            num_df = dataframe.select_dtypes(include='number')
            if response_var in num_df.columns:
                corr_with_response = num_df.corr()[response_var]
                for var in corr_with_response.index:
                    if var in summary_df['Variable'].values:
                        summary_df.loc[summary_df['Variable'] == var, 'Correlation with Response'] = corr_with_response[
                            var]

    if verbose:
        print_section("Shape of the dataframe")
        print(f"Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}")

        print_section("Column Classification")
        print(f"Categorical Columns: {cat_cols}\n")
        print(f"Numerical Columns: {num_cols}\n")
        print(f"Cardinal Categorical Columns: {cat_but_car}\n")
        print(f"Date Columns: {date_cols}\n")

        print_section("Unique Values, Memory Usage, Null and Zero Value Ratios")

        print_section("Head")
        print(dataframe.head(head))

        print_section("Tail")
        print(dataframe.tail(head))

        print_section("Missing Values")
        missing = dataframe.isnull().sum()
        missing_pct = 100 * dataframe.isnull().sum() / len(dataframe)
        missing_table = pd.concat([missing, missing_pct], axis=1, keys=['Total', 'Percent'])

        missing_columns = missing_table[missing_table['Total'] > 0].sort_values('Total', ascending=False)
        num_missing = len(missing_columns)
        num_cols_pie = 5
        num_rows = -(-num_missing // num_cols_pie)

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols_pie,
            specs=[[{'type': 'domain'}] * num_cols_pie for _ in range(num_rows)],
            subplot_titles=missing_columns.index.tolist()
        )

        for i, col in enumerate(missing_columns.index):
            row = i // num_cols_pie + 1
            col_idx = i % num_cols_pie + 1

            missing_count = missing[col]
            non_missing_count = len(dataframe) - missing_count
            total_count = missing_count + non_missing_count
            pie_data = [missing_count, non_missing_count]
            pie_labels = ['Missing', 'Not Missing']
            pie_percentages = [100 * missing_count / total_count, 100 * non_missing_count / total_count]

            fig.add_trace(
                go.Pie(
                    labels=pie_labels,
                    values=pie_data,
                    hole=0.4,
                    marker=dict(colors=['#ff6f6f', '#6f9eaf']),
                    textinfo='label+percent',
                    name=col
                ),
                row=row,
                col=col_idx
            )

        fig.update_layout(
            title_text="Missing Values - Pie Charts",
            showlegend=True,
            legend=dict(
                x=1.05,
                y=1.0,
                traceorder='normal',
                orientation='v'
            ),
            height=num_rows * 300,
            width=num_cols_pie * 300
        )

        fig.show()

        print_section('Mean Values of Response Variable by Categorical Variables')
        n_cols = 4
        n_rows = (len(cat_cols) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols,
                            subplot_titles=[f'Mean SalePrice by {col}' for col in cat_cols])
        for i, col in enumerate(cat_cols):
            df_filtered = df.groupby(col)[response_var].mean().reset_index()
            fig.add_trace(
                go.Bar(x=df_filtered[col], y=df_filtered[response_var], name='SalePrice', showlegend=False),
                row=i // n_cols + 1, col=i % n_cols + 1
            )
            mean_value = df_filtered[response_var].mean()
            fig.add_trace(
                go.Scatter(x=[df_filtered[col].min(), df_filtered[col].max()], y=[mean_value, mean_value],
                           mode='lines', line=dict(color='red', dash='dash'), name='Mean SalePrice', showlegend=False),
                row=i // n_cols + 1, col=i % n_cols + 1
            )
        fig.update_layout(
            height=300 * n_rows,
            width=1500,
            title_text="Mean SalePrice by Categorical Variables",
            title_x=0.5
        )
        fig.show()

        print_section("Descriptive Statistics")
        datetime_columns = dataframe.select_dtypes(include='datetime64[ns]').columns
        desc = dataframe.drop(datetime_columns, axis=1).describe([0, q1, q2, q3, q4, 1]).T
        desc['range'] = desc['max'] - desc['min']
        desc['coef_var'] = desc['std'] / desc['mean']
        print(desc)

        print_section("Correlation Heatmap and Correlations with Response Variable")
        # plt.figure(figsize=(24, 20))
        corr = dataframe.select_dtypes(include='number').corr()
        mask = np.triu(np.ones_like(corr, dtype='bool'))
        plt.figure(figsize=(24, 20))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f')
    else:
        return summary_df


df_coordinates_county = df.groupby('county')[['latitude', 'longitude']].mean().reset_index()


def create_county_markers(lats, lons, counties, location=[62.0, 15.0], zoom_start=4, width=800, height=600):
    """
    Create a map with circle markers for given counties and their coordinates.

    Parameters:
    - lats (list or pd.Series): List or Series of latitudes.
    - lons (list or pd.Series): List or Series of longitudes.
    - counties (list or pd.Series): List or Series of county names.
    - location (list of float): Initial map center coordinates [latitude, longitude]. Default is [62.0, 15.0].
    - zoom_start (int): Initial map zoom level. Default is 4.

    Returns:
    - sweden_map (folium.Map): Folium map object with the county markers.
    """

    if not isinstance(lats, (list, pd.Series)):
        raise ValueError("Lats must be a list or a pandas Series")
    if not isinstance(lons, (list, pd.Series)):
        raise ValueError("Lons must be a list or a pandas Series")
    if not isinstance(counties, (list, pd.Series)):
        raise ValueError("Counties must be a list or a pandas Series")
    if len(lats) != len(lons) or len(lats) != len(counties):
        raise ValueError("Lats, lons, and counties must have the same length")

    if isinstance(lats, pd.Series):
        lats = lats.tolist()
    if isinstance(lons, pd.Series):
        lons = lons.tolist()
    if isinstance(counties, pd.Series):
        counties = counties.tolist()

    sweden_map = folium.Map(location=location, zoom_start=zoom_start, width=width, height=height)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue',
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    for county, lat, lon in zip(counties, lats, lons):
        color = random.choice(colors)
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            popup=f"County: {county},  Lat: {lat}, Lon: {lon}",
            color=color,
            fill=True,
            fillColor=color
        ).add_to(sweden_map)
    return sweden_map


def create_price_heatmap(dataframe):
    """
    Create an interactive heatmap based on price values using given coordinates grouped by county.
    Includes a mouse position display showing coordinates, markers for the top 3 most expensive counties,
    and legends for both the markers and the heatmap colors.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing 'county', 'latitude', 'longitude', and 'price' columns.

    Returns:
    - folium.Map: Folium map object with the heatmap, mouse position display, markers, and legends.
    """

    if not all(col in dataframe.columns for col in ['county', 'latitude', 'longitude', 'price']):
        raise ValueError("DataFrame must contain 'county', 'latitude', 'longitude', and 'price' columns")

    map_center = [dataframe['latitude'].mean(), dataframe['longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=6)

    heat_data = dataframe[['latitude', 'longitude', 'price']].values.tolist()
    HeatMap(heat_data).add_to(folium_map)

    formatter = "function(num) {return L.Util.formatNum(num, 5) + ' º ';};"
    mouse_position = MousePosition(
        position='topright',
        separator=' | ',
        empty_string='NaN',
        lng_first=True,
        num_digits=20,
        prefix="Coordinates:",
        lat_formatter=formatter,
        lng_formatter=formatter
    )
    folium_map.add_child(mouse_position)

    top_3_counties = dataframe.groupby('county')['price'].mean().nlargest(3).reset_index()

    colors = ['red', 'darkred', 'orange']
    for idx, row in top_3_counties.iterrows():
        county_data = dataframe[dataframe['county'] == row['county']].iloc[0]
        folium.Marker(
            location=[county_data['latitude'], county_data['longitude']],
            popup=f"County: {row['county']}<br>Avg Price: ${row['price']:,.2f}",
            icon=folium.Icon(color=colors[idx], icon='star', prefix='fa')
        ).add_to(folium_map)

    legend_html = '''
     <div style="
     position: fixed;
     bottom: 50px; left: 50px; width: 220px;
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; padding: 10px;
     ">
     <strong>Top 3 Most Expensive Counties</strong><br>
     <i class="fa fa-star" style="color: red;"></i> 1st Most Expensive<br>
     <i class="fa fa-star" style="color: darkred;"></i> 2nd Most Expensive<br>
     <i class="fa fa-star" style="color: orange;"></i> 3rd Most Expensive<br>
     <br>
     <strong>Price Heatmap</strong><br>
     <i style="background: rgba(255, 0, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> High<br>
     <i style="background: rgba(255, 165, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Medium<br>
     <i style="background: rgba(0, 255, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Low<br>
     </div>
     '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))

    return folium_map


df_coordinates_county_price = df.groupby('county')[['latitude', 'longitude', 'price']].mean().sort_values(
    'price',
    ascending=False).reset_index()

legend_html = '''
 <div style="
 position: fixed;
 bottom: 50px; left: 50px; width: 220px;
 background-color: white; z-index:9999; font-size:14px;
 border:2px solid grey; padding: 10px;
 ">
 <strong>Top 3 Most Expensive Counties</strong><br>
 <i class="fa fa-star" style="color: red;"></i> 1st Most Expensive<br>
 <i class="fa fa-star" style="color: darkred;"></i> 2nd Most Expensive<br>
 <i class="fa fa-star" style="color: orange;"></i> 3rd Most Expensive<br>
 <br>
 <strong>Price Heatmap</strong><br>
 <i style="background: rgba(255, 0, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> High<br>
 <i style="background: rgba(255, 165, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Medium<br>
 <i style="background: rgba(0, 255, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Low<br>
 </div>
 '''


def create_house_count_heatmap(dataframe):
    """
    Create an interactive heatmap based on the number of houses using given coordinates grouped by county.
    Includes a mouse position display showing coordinates, markers for the top 3 most populated counties,
    and legends for both the markers and the heatmap colors.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing 'county', 'latitude', 'longitude', and 'price' columns.

    Returns:
    - folium.Map: Folium map object with the heatmap, mouse position display, markers, and legends.
    """

    if not all(col in dataframe.columns for col in ['county', 'latitude', 'longitude']):
        raise ValueError("DataFrame must contain 'county', 'latitude', 'longitude' columns")

    map_center = [dataframe['latitude'].mean(), dataframe['longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=6)

    # Count the number of houses in each coordinate
    heat_data = dataframe.groupby(['latitude', 'longitude']).size().reset_index(name='count').values.tolist()
    HeatMap(heat_data, radius=15).add_to(folium_map)

    formatter = "function(num) {return L.Util.formatNum(num, 5) + ' º ';};"
    mouse_position = MousePosition(
        position='topright',
        separator=' | ',
        empty_string='NaN',
        lng_first=True,
        num_digits=20,
        prefix="Coordinates:",
        lat_formatter=formatter,
        lng_formatter=formatter
    )
    folium_map.add_child(mouse_position)

    top_3_counties = dataframe['county'].value_counts().nlargest(3).reset_index()
    top_3_counties.columns = ['county', 'count']

    colors = ['red', 'darkred', 'orange']
    for idx, row in top_3_counties.iterrows():
        county_data = dataframe[dataframe['county'] == row['county']].iloc[0]
        folium.Marker(
            location=[county_data['latitude'], county_data['longitude']],
            popup=f"County: {row['county']}<br>Number of Houses: {row['count']}",
            icon=folium.Icon(color=colors[idx], icon='star', prefix='fa')
        ).add_to(folium_map)

    legend_html = '''
     <div style="
     position: fixed;
     bottom: 50px; left: 50px; width: 220px;
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; padding: 10px;
     ">
     <strong>Top 3 Most Populated Counties</strong><br>
     <i class="fa fa-star" style="color: red;"></i> 1st Most Populated<br>
     <i class="fa fa-star" style="color: darkred;"></i> 2nd Most Populated<br>
     <i class="fa fa-star" style="color: orange;"></i> 3rd Most Populated<br>
     <br>
     <strong>House Count Heatmap</strong><br>
     <i style="background: rgba(255, 0, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> High<br>
     <i style="background: rgba(255, 165, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Medium<br>
     <i style="background: rgba(0, 255, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Low<br>
     </div>
     '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))

    return folium_map

with st.sidebar:
    st.title('The Sections')
    section = st.selectbox(
        'Select a Section',
        ['Home',
         'Data Description',
         'First Look and EDA',
         'Deciding Best Model',
         'Prediction',
         'Recommendation']
    )

if section == 'Home':
    st.markdown('<h1>Välkommen</h1>', unsafe_allow_html=True)
    st.image('https://www.crossed-flag-pins.com/animated-flag-gif/gifs/Sweden_240-animated-flag-gifs.gif')
    st.markdown('<h2>Sweden House Price Prediction and House Recommendation</h2>', unsafe_allow_html=True)
    st.markdown(
        '<h4>Welcome to the House Price Prediction app! Explore and predict house prices based on various parameters.</h4>',
        unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        video_path = r"C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/Swedish National Anthem - 'Du Gamla, Du Fria' (SV-EN).mp4"
        st.video(video_path)

    with col2:
        st.image("https://www.earsel.org/symposia/2015-symposium-Stockholm/images/sthlm.gif", width=600)

elif section == 'Data Description':
    st.markdown(
        """
        <style>

        .data-description {
            background-color: #c8e6c9;  /* Slightly darker green */
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title of the app
    st.title("Housing Data Variables")

    # Table of variables and descriptions
    data_description = {
        'street': 'Street name or address',
        'build_year': 'Year of building construction',
        'ownership_type': 'Type of ownership (e.g., owned, rented)',
        'housing_form': 'Type of housing (e.g., apartment, house)',
        'living_area': 'Living area in square meters',
        'land_area': 'Land area in square meters',
        'county': 'County where the property is located',
        'area': 'Area of the property',
        'price': 'Price of the property',
        'wanted_price': 'Price the seller wants',
        'latitude': 'Latitude of the property',
        'longitude': 'Longitude of the property',
        'fee': 'Monthly fee (if applicable)',
        'operating_cost': 'Operating costs of the property',
        'rooms': 'Number of rooms',
        'floor': 'Floor number of the property',
        'balcony': 'Presence of a balcony (yes/no)',
        'construction_date': 'Date of construction',
        'association': 'Property association details',
        'estate_agent': 'Name of the estate agent',
        'url': 'URL for more information about the property',
        'price_change': 'Change in price over time',
        'story': 'Story or history of the property',
        'sold_year': 'Year when the property was sold',
        'sold_month': 'Month when the property was sold',
        'sold_date': 'Exact date when the property was sold',
        'year_difference': 'Difference in years between sale and current date',
        'Renewed': 'Indicates if the property has been renewed'
    }

    st.write("### Variable Descriptions")
    for key, value in data_description.items():
        st.markdown(f"<div class='data-description'><strong>{key}:</strong> {value}</div>", unsafe_allow_html=True)

if section == 'First Look and EDA':
    if 'eda_view' not in st.session_state:
        st.session_state.eda_view = 'Before'

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('Before'):
            st.session_state.eda_view = 'Before'
    with col2:
        if st.button('After'):
            st.session_state.eda_view = 'After'
    if st.session_state.eda_view == 'Before':
        tab1, tab2, tab3, tab4 = st.tabs(['General Overview', 'Value Counts Ratio of Categorical Columns', 'Null Values', 'Zero Values'])
        with tab1:
            summary_df = pd.read_csv('summary_df.csv')
            st.dataframe(summary_df)
        with tab2:
            categorical_columns = st.selectbox(
                'Categorical Columns',
                ['Housing Form', 'Ownership Type','Number of Rooms','Presence of Balcony']
            )
            if categorical_columns == 'Housing Form':
                st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/housing_form_before.png")
            elif categorical_columns == 'Ownership Type':
                st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/ownership_type_before.png")
            elif categorical_columns == 'Number of Rooms':
                st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/rooms_before.png")
            elif categorical_columns == 'Presence of Balcony':
                st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/balcony_before.png")
        with tab3:
            st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/missing_values_before.png")
        with tab4:
            st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/zero_values_before.png")
    if st.session_state.eda_view == 'After':
        tab1, tab2 = st.tabs(['EDA Process Flow', 'Clustering'])
        with tab1:
            nodes = [StreamlitFlowNode(id='1', pos=(0, 200), data={'content': 'Transforming variables-feature engineering'}, node_type='input',
                                       source_position='right', style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                                         'width': '50px', 'height': '50px'}),
                     StreamlitFlowNode('2', (350, 0), {'content': 'sold_at column'}, 'default', 'right', 'left',
                                       style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('3', (350, 150), {'content': 'Revised(if build_date>sold_date'}, 'default', 'right', 'left'),
                     StreamlitFlowNode('4', (600, 0), {'content': 'sold_day, sold_month, sold_year'}, 'default','bottom',
                                       'left',style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('5', (600, 150), {'content': 'sold_date'}, 'default', 'left', 'top',
                                       style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('6', (350, 300), {'content': 'clustering coordinates'}, 'default', 'right', 'left',
                                       style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('7', (600, 300), {'content': 'creating optimum Sweden regions'}, 'default', target_position='left',
                                       style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('8', (300, 400), {'content': 'creating new numerical features'}, 'default', 'right',
                                       'left', style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('9', (600, 400), {'content': 'rooms/living area, wanted_price*price_change etc.'}, 'default',
                                       'right', 'left', style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('10', (900, 400), {'content': 'if multicollinearity ---- > drop'}, 'output',target_position='left',
                                       style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                                'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('11', (0, 600), {'content': 'dealing with null and zero values'}, 'input', 'right',
                                       style={'color': 'white', 'backgroundColor': '#8B0000',
                                              'border': '2px solid white',
                                              'width': '50px', 'height': '50px'}),
                     StreamlitFlowNode('12', (300, 500), {'content' : 'if too many missing values ----> drop'}, 'default',
                                       'right', 'left', style = {'color': 'white', 'backgroundColor': '#8B0000',
                                              'border': '2px solid white',
                                              'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('13', (600, 500), {'content': 'drop the construction date and operation type'}, 'output',
                                       target_position= 'left', style = {'color': 'white', 'backgroundColor': '#8B0000',
                                              'border': '2px solid white',
                                              'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('14', (300, 700), {'content' : 'filling null and zero with categorical variables'}, 'default',
                                       'right', 'left', sytle = {'color': 'white', 'backgroundColor': '#8B0000',
                                              'border': '2px solid white',
                                              'width': '30px', 'height': '30px'}),
                     StreamlitFlowNode('15', (600, 700), {'content' : 'price_change, area, wanted_price etc. filled'}, 'output',
                                       target_position= 'left', style = {'color': 'white', 'backgroundColor': '#8B0000',
                                              'border': '2px solid white',
                                              'width': '30px', 'height': '30px'})]



            edges = [StreamlitFlowEdge('1-2', '1', '2', animated=True),
                     StreamlitFlowEdge('1-3', '1', '3', animated=True),
                     StreamlitFlowEdge('2-4', '2', '4', animated=True),
                     StreamlitFlowEdge('4-5', '4', '5', animated = True),
                     StreamlitFlowEdge('5-3', '3', '5', animated= True),
                     StreamlitFlowEdge('1-6', '1', '6', animated = True),
                     StreamlitFlowEdge('6-7', '6', '7', animated = True),
                     StreamlitFlowEdge('1-8', '1', '8', animated = True),
                     StreamlitFlowEdge('8-9', '8', '9', animated = True),
                     StreamlitFlowEdge('9-10', '9', '10', animated = True),
                     StreamlitFlowEdge('11-12', '11', '12', animated = True),
                     StreamlitFlowEdge('12-13', '12', '13', animated= True),
                     StreamlitFlowEdge('11-14', '11', '14', animated = True),
                     StreamlitFlowEdge('14-15', '14', '15', animated = True)]

            streamlit_flow('static_flow',
                           nodes,
                           edges,
                           fit_view=True,
                           show_minimap=False,
                           show_controls=False,
                           pan_on_drag=True,
                           allow_zoom=True)
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.title('The Elbow Plot')
                st.image("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/elbow_plot.png")
            with col2:
                def display_map_in_streamlit(map_file):
                    with open(map_file, 'r') as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, height=400)
                st.title('Clustering with 10000 samples')
                display_map_in_streamlit("C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/cluster.html")

#"C:/Users/oguzh/OneDrive/Masaüstü/TAKE_ME_TO_THE_HOUSE/cluster.html"
    # # Display content based on the selected EDA view
    # if st.session_state.eda_view == 'Before':
    #     st.header("Before View")
    #     tab1, tab2, tab3 = st.tabs(['General Overview', 'Value Counts Ratio of Categorical Columns', 'Null Values'])
    #
    #     with tab1:
    #         summary_df = dataframe_summary(df2, response_var='price')
    #         st.dataframe(summary_df)
    #
    #     with tab2:
    #         cat_cols = [col for col in df2.select_dtypes(include=['object']).columns if col != 'price']
    #         n_cols = 4
    #         n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    #         fig = make_subplots(rows=n_rows, cols=n_cols,
    #                             subplot_titles=[f'Mean SalePrice by {col}' for col in cat_cols])
    #         for i, col in enumerate(cat_cols):
    #             df_filtered = df2.groupby(col)['price'].mean().reset_index()
    #             fig.add_trace(
    #                 go.Bar(x=df_filtered[col], y=df_filtered["price"], name='SalePrice', showlegend=False),
    #                 row=i // n_cols + 1, col=i % n_cols + 1
    #             )
    #             mean_value = df_filtered['price'].mean()
    #             fig.add_trace(
    #                 go.Scatter(x=[df_filtered[col].min(), df_filtered[col].max()], y=[mean_value, mean_value],
    #                            mode='lines', line=dict(color='red', dash='dash'), name='Mean SalePrice',
    #                            showlegend=False),
    #                 row=i // n_cols + 1, col=i % n_cols + 1
    #             )
    #         fig.update_layout(
    #             height=300 * n_rows,
    #             width=1500,
    #             title_text="Mean SalePrice by Categorical Variables",
    #             title_x=0.5
    #         )
    #         st.plotly_chart(fig)
    #
    #     with tab3:
    #         missing = df2.isnull().sum()
    #         missing_pct = 100 * df2.isnull().sum() / len(df2)
    #         missing_table = pd.concat([missing, missing_pct], axis=1, keys=['Total', 'Percent'])
    #
    #         missing_columns = missing_table[missing_table['Total'] > 0].sort_values('Total', ascending=False)
    #         num_missing = len(missing_columns)
    #         num_cols_pie = 5
    #         num_rows = -(-num_missing // num_cols_pie)
    #
    #         fig = make_subplots(
    #             rows=num_rows,
    #             cols=num_cols_pie,
    #             specs=[[{'type': 'domain'}] * num_cols_pie for _ in range(num_rows)],
    #             subplot_titles=missing_columns.index.tolist()
    #         )
    #
    #         for i, col in enumerate(missing_columns.index):
    #             row = i // num_cols_pie + 1
    #             col_idx = i % num_cols_pie + 1
    #
    #             missing_count = missing[col]
    #             non_missing_count = len(df) - missing_count
    #             total_count = missing_count + non_missing_count
    #             pie_data = [missing_count, non_missing_count]
    #             pie_labels = ['Missing', 'Not Missing']
    #             pie_percentages = [100 * missing_count / total_count, 100 * non_missing_count / total_count]
    #
    #             fig.add_trace(
    #                 go.Pie(
    #                     labels=pie_labels,
    #                     values=pie_data,
    #                     hole=0.4,
    #                     marker=dict(colors=['#ff6f6f', '#6f9eaf']),
    #                     textinfo='label+percent',
    #                     name=col
    #                 ),
    #                 row=row,
    #                 col=col_idx
    #             )
    #
    #         fig.update_layout(
    #             title_text="Missing Values - Pie Charts",
    #             showlegend=True,
    #             legend=dict(
    #                 x=1.05,
    #                 y=1.0,
    #                 traceorder='normal',
    #                 orientation='v'
    #             ),
    #             height=num_rows * 300,
    #             width=num_cols_pie * 300
    #         )
    #         st.plotly_chart(fig)
    #
    # elif st.session_state.eda_view == 'After':
    #     st.header("After View")
    #     tab1, tab2, tab3 = st.tabs(['General Overview', 'Value Counts Ratio of Categorical Columns', 'Null Values'])
    #
    #     with tab1:
    #         summary_df = dataframe_summary(df, response_var='price')
    #         st.dataframe(summary_df)
    #
    #     with tab2:
    #         cat_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'price']
    #         n_cols = 4
    #         n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    #         fig = make_subplots(rows=n_rows, cols=n_cols,
    #                             subplot_titles=[f'Mean SalePrice by {col}' for col in cat_cols])
    #         for i, col in enumerate(cat_cols):
    #             df_filtered = df.groupby(col)['price'].mean().reset_index()
    #             fig.add_trace(
    #                 go.Bar(x=df_filtered[col], y=df_filtered["price"], name='SalePrice', showlegend=False),
    #                 row=i // n_cols + 1, col=i % n_cols + 1
    #             )
    #             mean_value = df_filtered['price'].mean()
    #             fig.add_trace(
    #                 go.Scatter(x=[df_filtered[col].min(), df_filtered[col].max()], y=[mean_value, mean_value],
    #                            mode='lines', line=dict(color='red', dash='dash'), name='Mean SalePrice',
    #                            showlegend=False),
    #                 row=i // n_cols + 1, col=i % n_cols + 1
    #             )
    #         fig.update_layout(
    #             height=300 * n_rows,
    #             width=1500,
    #             title_text="Mean SalePrice by Categorical Variables",
    #             title_x=0.5
    #         )
    #         st.plotly_chart(fig)
    #
    #     with tab3:
    #         missing = df.isnull().sum()
    #         missing_pct = 100 * df.isnull().sum() / len(df)
    #         missing_table = pd.concat([missing, missing_pct], axis=1, keys=['Total', 'Percent'])
    #
    #         missing_columns = missing_table[missing_table['Total'] > 0].sort_values('Total', ascending=False)
    #         num_missing = len(missing_columns)
    #         num_cols_pie = 5
    #         num_rows = -(-num_missing // num_cols_pie)
    #
    #         fig = make_subplots(
    #             rows=num_rows,
    #             cols=num_cols_pie,
    #             specs=[[{'type': 'domain'}] * num_cols_pie for _ in range(num_rows)],
    #             subplot_titles=missing_columns.index.tolist()
    #         )
    #
    #         for i, col in enumerate(missing_columns.index):
    #             row = i // num_cols_pie + 1
    #             col_idx = i % num_cols_pie + 1
    #
    #             missing_count = missing[col]
    #             non_missing_count = len(df) - missing_count
    #             total_count = missing_count + non_missing_count
    #             pie_data = [missing_count, non_missing_count]
    #             pie_labels = ['Missing', 'Not Missing']
    #             pie_percentages = [100 * missing_count / total_count, 100 * non_missing_count / total_count]
    #
    #             fig.add_trace(
    #                 go.Pie(
    #                     labels=pie_labels,
    #                     values=pie_data,
    #                     hole=0.4,
    #                     marker=dict(colors=['#ff6f6f', '#6f9eaf']),
    #                     textinfo='label+percent',
    #                     name=col
    #                 ),
    #                 row=row,
    #                 col=col_idx
    #             )
    #
    #         fig.update_layout(
    #             title_text="Missing Values - Pie Charts",
    #             showlegend=True,
    #             legend=dict(
    #                 x=1.05,
    #                 y=1.0,
    #                 traceorder='normal',
    #                 orientation='v'
    #             ),
    #             height=num_rows * 300,
    #             width=num_cols_pie * 300
    #         )
    #         st.plotly_chart(fig)



elif section == 'Deciding Best Model':
    tab1, tab2, tab3 = st.tabs(['Model Metrics', 'Shap Diagrams', 'BeeSwarm Diagrams'])
    with tab1:
        metrics_df = pd.read_csv('metrics.csv')
        st.table(metrics_df)
    with tab2:
        model = st.selectbox('Models', ['Catboost', 'LightGBM', 'XGBoost'])
        if model == 'Catboost':
            st.image('shap_catboost.png')
        elif model == 'LightGBM':
            st.image('shap_lgb.png')
        elif model == 'XGBoost':
            st.image('shap_xgb.png')
    with tab3:
        model = st.selectbox('Models', ['Catboost', 'XGBoost'])
        if model == 'Catboost':
            st.image('shap_beeswarm_cb.png')
        elif model == 'XGBoost':
            st.image('shap_beeswarm_xgb.png')

elif section == 'Prediction':
    tab1, tab2 = st.tabs(['Prediction Flow', 'Results'])
    with tab1:
        st.markdown('<h2>Testing the models with predicted categorical column</h2>', unsafe_allow_html=True)
        nodes = [StreamlitFlowNode(id='1', pos=(0, 400), data={'content': 'How good our model predict after predict a categorical column with a classification model'},
                                   node_type='input',
                                   source_position='bottom',
                                   style={'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                          'width': '50px', 'height': '50px'}),
                 StreamlitFlowNode('2', (0, 600), {'content' : 'decide a train and test ratio for balcony response variable and test a model in main data'},
                                   'default', 'bottom', 'top',
                                   style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                          'width': '50px', 'height': '50px'}),
                 StreamlitFlowNode('3', (0, 800), {'content': 'choose best model'}, 'default', 'right', 'top',
                                   style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                          'width': '50px', 'height': '50px'}),
                 StreamlitFlowNode('4', (300, 800), {'content' : 'obtain the sample data from where balcony is null the size of %.25 of the main data(test_size)'},
                                   'default', 'right', 'left',
                                   style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                          'width': '50px', 'height': '50px'}),
                 StreamlitFlowNode('5', (600, 800), {'content' : 'predict the balcony variable with best model'}, 'default',
                                   'right', 'left',
                                   style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                          'width': '50px', 'height': '50px'}),
                 StreamlitFlowNode('6', (900, 800), {'content' : 'predict price and with multiple models and compare them the actual price'},
                                   'output', target_position='left',
                                   style = {'color': 'white', 'backgroundColor': '#006400', 'border': '2px solid white',
                                          'width': '50px', 'height': '50px'})]
        edges = [StreamlitFlowEdge('1-2', '1', '2', animated=True),
                 StreamlitFlowEdge('2-3', '2', '3', animated=True),
                 StreamlitFlowEdge('3-4', '3', '4', animated=True),
                 StreamlitFlowEdge('4-5', '4', '5', animated=True),
                 StreamlitFlowEdge('5-6', '5', '6', animated = True)]

        streamlit_flow('static_flow',
                       nodes,
                       edges,
                       fit_view=True,
                       show_minimap=False,
                       show_controls=False,
                       pan_on_drag=True,
                       allow_zoom=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h2>Classification Results</h2>', unsafe_allow_html=True)
            metrics = pd.read_csv('metrics_df_class.csv')
            st.table(metrics)
            st.markdown('<h2>Confusion Matrix</h2>', unsafe_allow_html=True)
            st.image("heatmap.png")
        with col2:
            st.markdown('<h2>Predicted vs. Actual Price</h2>', unsafe_allow_html=True)
            st.image("regplot_lgb.jpeg")


elif section == 'Recommendation':
    def filter_properties(df, price_min, price_max, property_type, living_area_min, living_area_max, balcony, city,
                          rooms):
        filtered_df = df[
            (df['price'] >= price_min) &
            (df['price'] <= price_max) &
            (df['living_area'] >= living_area_min) &
            (df['living_area'] <= living_area_max) &
            (df['rooms'] >= rooms)
            ]

        if property_type:
            filtered_df = filtered_df[filtered_df['property_type'].str.contains(property_type, case=False)]

        if balcony == 'YES':
            filtered_df = filtered_df[filtered_df['balcony'] == 'Yes']
        elif balcony == 'NO':
            filtered_df = filtered_df[filtered_df['balcony'] != 'Yes']

        if city:
            filtered_df = filtered_df[filtered_df['county'] == city]

        return filtered_df


    # Streamlit uygulaması
    st.title("HOUSE RECOMMENDATION SYSTEMS")

    st.sidebar.header("Filterings")
    price_min = st.sidebar.number_input("Minimum fiyat", min_value=0.0, value=0.0, step=10000.0)
    price_max = st.sidebar.number_input("Maksimum fiyat", min_value=0.0, value=10000000.0, step=10000.0)

    # Mülk tipi için unique değerleri al
    property_types = ['Any'] + sorted(df['property_type'].unique())
    property_type = st.sidebar.selectbox("Property Type", property_types, index=0)

    living_area_min = st.sidebar.number_input("Minimum m2", min_value=0.0, value=0.0, step=1.0)
    living_area_max = st.sidebar.number_input("Maximum m2", min_value=0.0, value=1000.0, step=1.0)
    balcony = st.sidebar.selectbox("Do you want a balcony?", ["Yes", "No", "It doesn't matter"], index=2)
    city = st.sidebar.selectbox("City", ['Any'] + sorted(population_df['county'].unique()))
    rooms = st.sidebar.number_input("Minimum Room Number", min_value=0.0, value=1.0, step=1.0)

    property_type = None if property_type == "Any" else property_type
    balcony = None if balcony == "It doesn't matter" else balcony
    city = None if city == "Any" else city

    filtered_df = filter_properties(df, price_min, price_max, property_type, living_area_min, living_area_max, balcony,
                                    city, rooms)

    # İlk 10 evi seç
    top_10_df = filtered_df.head(10)

    st.header("Recommended Houses")
    st.write(top_10_df)

    # Fiyat Dağılım Grafiği
    st.header("Price Distribution Chart")
    price_chart = alt.Chart(top_10_df).mark_bar(strokeWidth=20).encode(
        x='price:Q',
        y='count()'
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(price_chart)

    # Harita ve Heatmap ekleme
    # Harita ve Heatmap ekleme
    st.header("Houses on the map and Price Heatmap")

    col1, col2 = st.columns(2)

    with col1:
        map_data = top_10_df[['latitude', 'longitude', 'area']]
        m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=10)
        for idx, row in map_data.iterrows():
            folium.Marker([row['latitude'], row['longitude']], popup=row['area']).add_to(m)
        st_folium(m)

    with col2:
        heatmap_data = filtered_df[['latitude', 'longitude', 'price']].dropna()
        heatmap_list = heatmap_data.values.tolist()

        hm = folium.Map(location=[heatmap_data['latitude'].mean(), heatmap_data['longitude'].mean()], zoom_start=10)
        plugins.HeatMap(heatmap_list).add_to(hm)

        st_folium(hm)

    # Seçilen şehir için popülasyon grafiği
    if city:
        city_population = population_df[population_df['county'].str.contains(city, case=False)]

        if not city_population.empty:
            color_sequence = ['#a3c6c4', '#d9e2e9', '#006400', '#a6d9f7']

            fig_city = px.bar(
                city_population,
                x='age_range',
                y='population',
                color='gender',
                color_discrete_sequence=color_sequence,
                title=f'Population distribution of {city} <br> by age groups',
                labels={'age_range': 'age_range', 'population': 'population'}
            )

            fig_city.update_layout(
                width=1200,
                height=720,
                barmode='group'
            )

    # İsveç'in genel popülasyon grafiği
    color_sequence = ['#a3c6c4', '#d9e2e9', '#006400', '#a6d9f7']

    fig = px.bar(
        population_df,
        x='age_range',
        y='population',
        color='gender',
        color_discrete_sequence=color_sequence,
        title='Population distribution of Sweden <br> by age groups',
        labels={'age_range': 'age_range', 'population': 'population'}
    )

    fig.update_layout(
        width=1200,
        height=720,
        barmode='group'
    )

    # Grafikleri yan yana gösterme
    col1, col2 = st.columns(2)

    with col1:
        if city and not city_population.empty:
            st.plotly_chart(fig_city)
        else:
            st.write("Seçilen şehir için popülasyon verisi mevcut değil.")

    with col2:
        st.plotly_chart(fig)

    # İsveç'in genel popülasyon grafiği
    total_population_df = population_df.groupby('county')['population'].sum().reset_index()

    st.header("General Population Distribution of Sweden")
    fig_sweden = px.bar(
        total_population_df,
        x='county',
        y='population',
        title='Total population distribution of Sweden <br> by cities',
        labels={'county': 'City', 'population': 'Total Population'}
    )

    fig_sweden.update_layout(
        width=1200,
        height=720
    )

    st.plotly_chart(fig_sweden)

    # education grafik çizme

    # education grafik çizme

    st.header("Education Rate by Age Group")

    if city:
        df_filtered = education_df[education_df['county'] == city]
        fig_edu = px.bar(
            df_filtered,
            x='age_range',
            y='population',
            title=f'Education rate of {city} by age groups',
            color_discrete_sequence=px.colors.sequential.Plasma_r  # Renk burada değiştirilir
        )
        st.plotly_chart(fig_edu)

        # Seçilen şehir için toplu taşıma seferleri grafiği

    if city:
        city_transport = transport_df[transport_df['County'].str.contains(city, case=False)]
        if not city_transport.empty:
            city_transport = city_transport.melt(id_vars=['County', 'Type'], var_name='Year', value_name='Trips')
            fig_transport = px.bar(
                city_transport,
                x='Year',
                y='Trips',
                color='Type',
                title=f'Transportation trips in {city} by year and type'
            )
            st.plotly_chart(fig_transport)
    #image_path = "pie_chart.jpg"


    # Display the green space distribution image
    st.header("Green Space Distribution by County")
    st.image("pie_chart.jpg", caption="Green Space Distribution by County")

    # CSV İndirme
    csv = top_10_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download recommended homes as CSV",
        data=csv,
        file_name='filtered_properties.csv',
        mime='text/csv',
    )















