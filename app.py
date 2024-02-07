from flask import Flask, request, render_template, Response
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

# Load your trained machine learning model
model_path = r"C:\Users\HunterAK\Digicrome\NextHike\Project 2\linear_regression_model_20240121145455.pkl"
model = pickle.load(open(model_path, 'rb'))

def adjust_date(input_date, days_to_add):
    return input_date + pd.DateOffset(days=days_to_add)

@app.route('/', methods=['GET'])
def predict():
    # Get input data from the request arguments
    store = int(request.args.get('store', 1))
    day_of_week = int(request.args.get('day_of_week', 1))
    open_status = int(request.args.get('open', 1))
    promo = int(request.args.get('promo', 0))
    state_holiday = int(request.args.get('state_holiday', 0))
    school_holiday = int(request.args.get('school_holiday', 0))

    # Continue converting other input features to their appropriate data types
    store_type = int(request.args.get('store_type', 1))
    assortment = int(request.args.get('assortment', 1))
    competition_distance = float(request.args.get('competition_distance', 500.0))
    competition_open_month = int(request.args.get('competition_open_month', 1))
    competition_open_year = int(request.args.get('competition_open_year', 2000))
    promo2 = int(request.args.get('promo2', 0))
    promo2_since_week = int(request.args.get('promo2_since_week', 1))
    promo2_since_year = int(request.args.get('promo2_since_year', 2000))
    promo_interval = request.args.get('promo_interval', "None")
    weekday = int(request.args.get('weekday', 0))
    is_weekend = int(request.args.get('is_weekend', 1))
    sales_per_customer = float(request.args.get('sales_per_customer', 0.0))
    season = int(request.args.get('season', 1))
    is_beginning_of_month = int(request.args.get('is_beginning_of_month', 0))
    is_mid_of_month = int(request.args.get('is_mid_of_month', 0))
    is_end_of_month = int(request.args.get('is_end_of_month', 0))
    days_to_holiday = int(request.args.get('days_to_holiday', 0))
    days_after_holiday = int(request.args.get('days_after_holiday', 0))

    # Handle promo_interval separately
    if promo_interval == "None":
        promo_interval = 0  
    else:
        promo_interval = float(promo_interval)

    # Create a list of selected feature names that match the features used during training
    selected_feature_names = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                              'StoreType', 'Assortment', 'CompetitionDistance',
                              'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                              'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'weekday',
                              'is_weekend', 'Season', 'IsBeginningOfMonth',
                              'IsMidOfMonth', 'IsEndOfMonth', 'DaysToHoliday', 'DaysAfterHoliday']

    # Create a dictionary to map feature names to their corresponding values
    input_features_dict = {
        'Store': store,
        'DayOfWeek': day_of_week,
        'Open': open_status,
        'Promo': promo,
        'StateHoliday': state_holiday,
        'SchoolHoliday': school_holiday,
        'StoreType': store_type,
        'Assortment': assortment,
        'CompetitionDistance': competition_distance,
        'CompetitionOpenSinceMonth': competition_open_month,
        'CompetitionOpenSinceYear': competition_open_year,
        'Promo2': promo2,
        'Promo2SinceWeek': promo2_since_week,
        'Promo2SinceYear': promo2_since_year,
        'PromoInterval': promo_interval,
        'weekday': weekday,
        'is_weekend': is_weekend,
        'SalesPerCustomer': sales_per_customer,  # Include your target feature
        'Season': season,
        'IsBeginningOfMonth': is_beginning_of_month,
        'IsMidOfMonth': is_mid_of_month,
        'IsEndOfMonth': is_end_of_month,
        'DaysToHoliday': days_to_holiday,
        'DaysAfterHoliday': days_after_holiday,
    }

    # Assuming "SalesPerCustomer" is your target value, remove it from input_features_dict
    input_features_dict.pop('SalesPerCustomer')

    # Create a dictionary to map feature names to their corresponding indices
    feature_indices = {feature: idx for idx, feature in enumerate(selected_feature_names)}

    # Create a numpy array to hold the input features
    input_features = np.zeros(len(selected_feature_names))

    # Populate the input_features array with values based on the feature indices
    for feature, value in input_features_dict.items():
        idx = feature_indices.get(feature)
        if idx is not None:
            input_features[idx] = value

    # Reshape the input_features for LSTM input (samples, time steps, features)
    input_features = input_features.reshape(1, 1, -1)

    # Initialize lists to store predictions for the next 6 months
    predictions_next_6_months = []

    # Define the start date for predictions (adjust as needed)
    start_date = pd.to_datetime('2023-09-01')

    # Loop to predict sales for the next 6 months
    for i in range(6):
        # Use your trained LSTM model to make predictions
        predicted_value = model.predict(input_features)[0][0]  
        predicted_value = float(predicted_value)

        # Add the prediction to the list
        predictions_next_6_months.append(predicted_value)

        # Update the input features for the next prediction (adjust date-related features)
        input_features[0][0][1] += 7  # Increment the day of the week by 7 days
        input_features[0][0][17] += 1  # Increment the month by 1
        input_features[0][0][19] += 1  # Increment the day of the month by 1

    # Create a list of dates for the next 6 months
    plot_dates = [start_date + pd.DateOffset(months=i) for i in range(6)]

    # Plot predictions
    plt.figure(figsize=(6, 4))  
    plt.plot(plot_dates, predictions_next_6_months, marker='o', linestyle='-', color='green')  # Change color to green
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('ROSSMAN SALES PREDICTION FOR NEXT 6 MONTHS')
    plt.grid(True)

    # Create a BytesIO buffer to hold the plot image
    plot_buffer = BytesIO()
    FigureCanvas(plt.gcf()).print_png(plot_buffer)
    plot_buffer.seek(0)
    
    # Save the plot to a file (change the filename as needed)
    plot_filename = 'sales_plot.png'
    plt.savefig(plot_filename)

    # Clear the plot to release resources
    plt.clf()
    plt.close()


    # Encode the plot image as base64
    plot_data_uri = base64.b64encode(plot_buffer.read()).decode('utf-8')

    return render_template('template.html', predictions=predictions_next_6_months, plot_data_uri=plot_data_uri)

if __name__ == '__main__':
    app.run(debug=True)
