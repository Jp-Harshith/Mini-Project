def predictionModel(n_days, stock_code):
    """
    Predict stock prices for the next `n_days` using SVR and return a Plotly figure.
    """
    # Importing libraries
    import yfinance as yf
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.svm import SVR
    from datetime import date, timedelta
    import numpy as np

    try:
        # Fetch data from Yahoo Finance for the last 6 months
        df = yf.download(stock_code, period="6mo")
        if df.empty:
            raise ValueError("No data found for the given stock ticker.")

        df.reset_index(inplace=True)
        df['Days'] = np.arange(len(df))  # Add a numerical index as 'Days'

        # Features and target variable
        X = df[['Days']].values
        y = df['Close'].values

        # Split the dataset (90% train, 10% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1],
            'gamma': ['scale', 'auto']
        }

        # Perform hyperparameter tuning with SVR
        gsc = GridSearchCV(
            estimator=SVR(kernel="rbf"),
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_absolute_error"
        )
        gsc.fit(X_train, y_train)

        # Best parameters from GridSearchCV
        best_params = gsc.best_params_
        svr_model = SVR(kernel="rbf", **best_params)
        svr_model.fit(X_train, y_train)

        # Prepare future dates for prediction
        last_day_index = X[-1][0]  # Last day in the training data
        future_days = np.arange(last_day_index + 1, last_day_index + 1 + n_days).reshape(-1, 1)

        # Predict future prices
        predicted_prices = svr_model.predict(future_days)

        # Generate future dates (skipping weekends)
        future_dates = []
        current_date = date.today()
        for _ in range(n_days):
            while current_date.weekday() >= 5:  # Skip Saturday and Sunday
                current_date += timedelta(days=1)
            future_dates.append(current_date)
            current_date += timedelta(days=1)

        # Plot results
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predicted_prices,
                mode="lines+markers",
                name="Predicted Prices"
            )
        )
        fig.update_layout(
            title=f"Predicted Close Price for Next {n_days} Days",
            xaxis_title="Date",
            yaxis_title="Close Price",
            template="plotly_white"
        )

        return fig

    except Exception as e:
        print(f"Error in predictionModel: {e}")
        return go.Figure().update_layout(title="Error generating prediction.")

