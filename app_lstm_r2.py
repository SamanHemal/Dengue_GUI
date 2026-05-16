from flask import Flask, render_template, request, session, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Bidirectional, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import os
import keras
import matplotlib.pyplot as plt
import matplotlib
import joblib
import json
import geopandas as gpd
from sklearn.metrics import mean_squared_error, mean_absolute_error

matplotlib.use('Agg')
app = Flask(__name__)
app.secret_key = 'dungue_gui'

# Load configurations from JSON file
with open('config.json') as config_file:
    config = json.load(config_file)

# Path to the folder where csv will be uploaded
UPLOAD_FOLDER = 'Uploads'

# Data preparation function
def prepare_data(df, features, scaler=None):
    df = df.copy()
    df['Time_Index'] = df.groupby(['District', 'Year']).cumcount()
    # df['Season'] = df['Month'] % 4// 3 + 1

    district_stats = df.groupby('District')[features].agg(['mean', 'std'])
    for feature in features:
        df[f'{feature}_district_z'] = df.groupby('District')[feature].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    if scaler is None:
        scaler = RobustScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])

    sequences = []
    targets = []
    months = 1
    for district in df['District'].unique():
        district_df = df[df['District'] == district]
        for i in range(months, len(district_df)):
            sequences.append(district_df[features].values[i-months:i])
            targets.append(district_df['D_Cases'].values[i])

    return np.array(sequences), np.array(targets), scaler

@app.route('/', methods=['GET', 'POST'])
def index():

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if 'csv' not in request.files:
        return render_template('index.html', error='No file found')

    file = request.files['csv']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        # Save the uploaded csv to the UPLOAD_FOLDER
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(csv_path)
        session['df_path'] = csv_path
    
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Store DataFrame in session as a dictionary
        #session['df_dict'] = pickle.dumps(df)
        # Convert the DataFrame to an HTML table
        html_table = df.to_html(classes='table table-striped', index=False)

        return render_template('index.html', result='CSV file processed successfully!', display_list=html_table)

    return render_template('index.html', result=None, display_list=None, error=None, months=None)

def custom_attention_deserializer(**kwargs):
    # If score_mode came in as a function object, force it back to 'dot'
    if 'score_mode' in kwargs and not isinstance(kwargs['score_mode'], str):
        kwargs['score_mode'] = 'dot'
        
    # Reconstruct the Attention layer using the cleaned parameters
    return keras.layers.Attention(**kwargs)

@app.route('/predict', methods=['POST'])
def predict():

    #csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Training_Data_Weekly_2024.csv')

    # Load the model and scaler
    model_path = '/workspaces/Dengue_Gui/content/saved_model/best_model.h5'
    scaler_path = '/workspaces/Dengue_Gui/content/saved_model/scaler.pkl'

    try:
      model = load_model(model_path, custom_objects={'Huber': Huber, 'Attention': custom_attention_deserializer})
      with open(scaler_path, 'rb') as f:
          scaler = joblib.load(f)
      print("Model and scaler loaded successfully.")

    except (FileNotFoundError, OSError):
      print("Model or scaler not found. Please train the model first.")
      exit()  # Exit if the model or scaler is not found
    
    # Retrieve the DataFrame from the session
    csv_path = session['df_path']
    if csv_path=='':
        return render_template('index.html', result='data file unavailable')
    else:
        data_df = pd.read_csv(csv_path)
    
    try:
      df = data_df.sort_values(['District', 'Year', 'Month', 'Week'])
    except FileNotFoundError:
      exit()

    features = ['T_min', 'T_max', 'H_min', 'H_max', 'R_tot',
                'Pop_Den', 'Tot_Area', 'In_Water', 'Forests', 'Land']

    # Convert the DataFrame to an HTML table
    html_table = df.to_html(classes='table table-striped', index=False)

    predicted_cases_arr = []
    predictions_arr = []
    actual_cases = []

    entered_districts = df['District'].unique().tolist()
    district_list = config["district_list"]  # Extract district list from config
    for district_id in entered_districts:

        # Filter the DataFrame for the current district
        district_df = df[df['District'] == district_id]

        # Prepare the new data
        X_new, y_new, _ = prepare_data(district_df, features, scaler)

        # Make predictions
        predictions = model.predict(X_new)
        predictions_arr.append(int(predictions[0][0]))
        predicted_cases_arr.append([district_list[str(district_id)], int(predictions[0][0])])
        actual_cases.append(district_df['D_Cases'].tolist()[4])
        
    mae = mean_absolute_error(actual_cases, predictions_arr)
    mse = mean_squared_error(actual_cases, predictions_arr)

    print(f"Prediction new data MAE: {mae:.4f}, MSE: {mse:.4f}")

    
    session['df_prediction']  = pd.DataFrame(predicted_cases_arr).to_dict()
    session['df_actual']  = pd.DataFrame(actual_cases).to_dict()


    return render_template('index.html', forecast=predicted_cases_arr, actual_case=actual_cases, display_list=html_table)

@app.route('/heatmap', methods=['POST'])

def heatmap():

    predicted_cases_hmap = pd.DataFrame(session['df_prediction'])
    actual_cases_hmap = pd.DataFrame(session['df_actual'])
    # Path to the directory containing shapefiles
    shapefile_path = config['shape_file_path']
    districts = gpd.read_file(shapefile_path)

    predicted_cases_list = predicted_cases_hmap.values.tolist()
    actual_cases_list = actual_cases_hmap.values.tolist()
    pre_data = transform_data(predicted_cases_list)
    act_data = transform_actual_data(actual_cases_list)
    pre_df = pd.DataFrame(pre_data, columns=['Predictions'])
    act_df = pd.DataFrame(act_data, columns=['Actual'])


    districts['dataP'] = pre_df['Predictions']
    districts['dataA'] = act_df['Actual']
    
    draw_image(districts,'dataP',config["save_filename_prediction"])
    draw_image(districts,'dataA',config["save_filename_actual"])

    predict_heatmap_url = f'/static/images/{config["save_filename_prediction"]}'
    actual_heatmap_url = f'/static/images/{config["save_filename_actual"]}'
    return jsonify({'predict_heatmap_url': predict_heatmap_url, 'actual_heatmap_url': actual_heatmap_url})

def transform_data(data):
    t_data = []
    t_data.extend([item[1] for item in data[0:7]])
    t_data.append(data[8][1])
    t_data.append(data[7][1])
    t_data.append(data[9][1])
    t_data.append(data[13][1])
    t_data.append(data[10][1])
    t_data.append(data[11][1])
    t_data.append(data[12][1])
    t_data.extend([item[1] for item in data[14:]])

    return t_data

def transform_actual_data(data):
    t_data = []
    t_data.extend([item for item in data[0:7]])
    t_data.append(data[8])
    t_data.append(data[7])
    t_data.append(data[9])
    t_data.append(data[13])
    t_data.append(data[10])
    t_data.append(data[11])
    t_data.append(data[12])
    t_data.extend([item for item in data[14:]])

    return t_data


def draw_image(districts,param,filename):
    # Plot each shapefile's boundaries
    fig, ax = plt.subplots(figsize=(3, 3))

    try:
        # Dissolve boundaries into single geometry for each district
        # district_boundaries = districts.dissolve()
        # Generate a colormap with unique colors for each district
        districts.plot(ax=ax, column=param,cmap='OrRd', legend=True,  legend_kwds={"orientation": config['legend_orientation']})
        districts.boundary.plot(ax=ax, color='black', linewidth=0.5)

    except ValueError as e:
        print(f"Error plotting {districts}: {e}")
    
    plt.axis('off')
    plt.tight_layout()

    heatmap_path = os.path.join(config['save_path'], filename)
    plt.savefig(heatmap_path, dpi=300)
    plt.close(fig)

@app.route('/permutation', methods=['POST'])

def permutation_importance():

    model_path = '/workspaces/Dengue_Gui/content/saved_model/best_model.h5'
    scaler_path = '/workspaces/Dengue_Gui/content/saved_model/scaler.pkl'

    try:
      model = load_model(model_path, custom_objects={'Huber': Huber, 'Attention': custom_attention_deserializer})
      with open(scaler_path, 'rb') as f:
          scaler = joblib.load(f)
      print("Model and scaler loaded successfully.")

    except (FileNotFoundError, OSError):
      print("Model or scaler not found. Please train the model first.")
      exit()  # Exit if the model or scaler is not found

    # Retrieve the DataFrame from the session
    csv_path = session['df_path']
    if csv_path=='':
        return render_template('index.html', result='data file unavailable')
    else:
        data_df = pd.read_csv(csv_path)
    
    try:
      df = data_df.sort_values(['District', 'Year', 'Month', 'Week'])
    except FileNotFoundError:
      exit()

    features = ['T_min', 'T_max', 'H_min', 'H_max', 'R_tot',
                'Pop_Den', 'Tot_Area', 'In_Water', 'Forests', 'Land']

    # Convert the DataFrame to an HTML table
    html_table = df.to_html(classes='table table-striped', index=False)

    entered_districts = df['District'].unique().tolist()
    district_list = config["district_list"]  # Extract district list from config
    district_importance = {}

    for district_id in entered_districts:

        # Filter the DataFrame for the current district
        district_df = df[df['District'] == district_id]

        # Prepare the new data
        X_new, y_new, _ = prepare_data(district_df, features, scaler)


        baseline_error = mean_squared_error(y_new, model.predict(X_new))
        importances = {}

        for i, feature in enumerate(features):
            X_permuted = X_new.copy()
            np.random.shuffle(X_permuted[:, :, i])  # Shuffle values of the feature across all samples
            error = mean_squared_error(y_new, model.predict(X_permuted))
            importances[feature] = error - baseline_error  # Importance is the increase in error

        # Normalize the importances
        total_importance = sum(importances.values())
        for feature in importances:
            if total_importance != 0:
                importances[feature] = importances[feature] / total_importance
            else:
                importances[feature] = 0
        
        district_importance[district_id] = importances

    return jsonify({'district_importance': district_importance, 'district_list': district_list})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)