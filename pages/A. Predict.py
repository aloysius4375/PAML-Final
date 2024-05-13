import streamlit as st
import pickle
import pandas as pd


# load models
def load_models(model_names):
    models = {}
    for name in model_names:
        try:
            with open(f'pkl/{name}.pkl', 'rb') as file:
                models[name] = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading {name} model: {e}")
    return models


# load encoder
def load_label_encoders():
    with open('pkl/label_encoders.pkl', 'rb') as f:
        return pickle.load(f)

label_encoders = load_label_encoders()


# collect inputs
def get_user_input(label_encoders):
    '''
    df should have the following features
    -float:
    'Temperature(F)', 'Wind_Chill(F)','Humidity(%)', 'Pressure(in)', 'Visibility(mi)‘,'Wind_Speed(mph)', 'Precipitation(in)'
    -String(total have 140+ choices):
    'Weather_Condition'
    -Boolean(True or False):
    'Amenity','Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
       'Turning_Loop'
    -String(Day or Night):
    'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight','Astronomical_Twilight'
    '''

    st.write("Please input the following environmental and road conditions:")
    
    st.write("#### 2.1 Weather Conditions")
    col1, col2 = st.columns(2)
    with col1:
        inputs = {
            'Weather_Condition': st.selectbox('Weather Condition', options=label_encoders['Weather_Condition'].classes_, help="Select the current weather conditions from the list."),
        }
    with col2:
        inputs['Precipitation(in)'] = st.number_input('Precipitation (in)', 0.0, 36.5, 0.0, 0.01, help="Enter the amount of precipitation in inches if any.")

    st.write("#### 2.2 Atmospheric Conditions")
    col3, col4 = st.columns(2)
    with col3:
        inputs['Temperature(F)'] = st.slider('Temperature (F)', -89.0, 207.0, 70.0, 0.1, help="Enter the current air temperature in degrees Fahrenheit.")
        inputs['Wind_Chill(F)'] = st.slider('Wind Chill (F)', -89.0, 207.0, 70.0, 0.1, help="Enter the perceived temperature considering the wind effect in degrees Fahrenheit.")
        inputs['Wind_Speed(mph)'] = st.slider('Wind Speed (mph)', 0.0, 1100.0, 10.0, 0.1, help="Enter the wind speed in miles per hour.")
    with col4:
        inputs['Humidity(%)'] = st.number_input('Humidity (%)', 1, 100, 50, 1, help="Enter the current air humidity in percentage.")
        inputs['Pressure(in)'] = st.number_input('Pressure (in)', 0.0, 60.0, 30.0, 0.01, help="Enter the atmospheric pressure in inches of mercury.")
        inputs['Visibility(mi)'] = st.number_input('Visibility (mi)', 0.0, 140.0, 10.0, 0.1, help="Enter the visibility distance in miles.")
  
    st.write("#### 2.3 Light Conditions")
    col5, col6 = st.columns(2)
    with col5:
        inputs['Sunrise_Sunset'] = st.selectbox('Sunrise/Sunset', ['Day', 'Night'], help="Choose whether it is day or night based on sunrise/sunset.")
        inputs['Civil_Twilight'] = st.selectbox('Civil Twilight', ['Day', 'Night'], help="Choose whether it is day or night based on civil twilight.")
    with col6:
        inputs['Nautical_Twilight'] = st.selectbox('Nautical Twilight', ['Day', 'Night'], help="Choose whether it is day or night based on nautical twilight.")
        inputs['Astronomical_Twilight'] = st.selectbox('Astronomical Twilight', ['Day', 'Night'], help="Choose whether it is day or night based on astronomical twilight.")

    st.write("#### 2.4 Traffic Features")
    inputs['Traffic_Features'] = st.multiselect('Select traffic-related features present:', 
        ['Amenity', 'Bump', 'Crossing', 'Give Way', 'Junction', 'No Exit', 'Railway', 
         'Roundabout', 'Station', 'Stop', 'Traffic Calming', 'Traffic Signal', 'Turning Loop'],
        help="Select any traffic-related features that are present at the location.")
    
    return inputs


def preprocess_input(input_data, label_encoders):
    df = pd.DataFrame([input_data])
    selected_traffic_features = input_data['Traffic_Features']
    TRAFFIC_FEATURES = [
    'Amenity', 'Bump', 'Crossing', 'Give Way', 'Junction', 'No Exit', 'Railway',
    'Roundabout', 'Station', 'Stop', 'Traffic Calming', 'Traffic Signal', 'Turning Loop'
    ]
    for feature in TRAFFIC_FEATURES:
        df[feature] = [feature in selected_traffic_features]  ## reassign boolean value for the 13 features using multiselect
    del df['Traffic_Features'] 
    for column, encoder in label_encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column].values)
    return df


def predict_risk(models, features):
    processed_features = preprocess_input(features, label_encoders)
    predictions = {}
    for model_name, model in models.items():
        try:
            # ensure the input is in the correct format (2D array)
            prediction = model.predict_proba(processed_features.values.reshape(1, -1))[0]
            predictions[model_name] = prediction
        except Exception as e:
            st.error(f"Error in making predictions with {model_name}: {e}")
    return predictions


##################################


st.title("Traffic Accident Risk Prediction")

st.write("## 1. Choose Your Model")
model_options = st.multiselect("#### Please select the models for prediction:", ["Random_Forest", "XGBoost", "AdaBoost"])
models = load_models(model_options)

st.write("## 2. Choose Your Inputs")
features = get_user_input(label_encoders)

st.write("## 3. Get Your Prediction Results")
if st.button('Predict Risk'):
    predictions = predict_risk(models, features)
    for model_name, prediction in predictions.items():
        st.write(f"Results for {model_name}: {prediction}")
    st.image("media/end.gif")

