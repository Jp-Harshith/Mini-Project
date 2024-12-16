# Importing libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime as dt, timedelta, date
from model import predictionModel  # Ensure this imports correctly

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# App Layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("Welcome to the Stock Trend Prediction App!", className="heading"),
                html.Div(
                    [
                        dcc.Input(
                            id='stock_code', value='', placeholder='Input Stock Ticker here',
                            type='text', className='inputs'
                        ),
                        html.Button('Submit', id='submit-stock', className='buttons', n_clicks=0)
                    ],
                    className=''
                ),
                html.Div(
                    [
                        dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=dt(1995, 8, 5),
                            max_date_allowed=dt.today(),
                            initial_visible_month=dt.now(),
                            start_date=date(2020, 1, 1),
                            className='inputs'
                        )
                    ],
                    className=''
                ),
                html.Div(
                    [
                        html.Button('Stock Price', id='stock_price', className='buttons'),
                        html.Button('Indicators', id='indicators', n_clicks=0, className='buttons'),
                    ],
                    className=''
                ),
                html.Div(
                    [
                        dcc.Input(
                            id='n_days', value='', type='text',
                            placeholder='Number of Days for Forecast', className='inputs'
                        ),
                        html.Button('Forecast', id='Forecast', className='buttons', n_clicks=0),
                        html.Button('Yearly Analysis', id='yearly_analysis', className='buttons', n_clicks=0),
                    ],
                    className=''
                ),
            ],
            className='nav'
        ),
        html.Div(
            [
                dcc.Loading(
                    id='loading1', color='#3b3b3b',
                    children=[html.Div(
                        [
                            html.Img(id='logo', className='imglogo'),
                            html.H2(id='ticker')
                        ],
                        className='header'
                    ),
                    html.Div(id='description', className='info') ]
                ),
                dcc.Loading(
                    children=[html.Div([], id='stonks-graph', className='graphs')],
                    id='loading2', type='graph'
                ),
                dcc.Loading(
                    id='loading3',
                    children=[html.Div([], id='forecast-graph', className='graphs')],
                    type='graph'
                ),
                dcc.Loading(
                    id='loading4',
                    children=[html.Div([], id='yearly-graph', className='graphs')],
                    type='graph'
                ),
            ],
            className='outputContainer'
        )
    ],
    className='container'
)

# Callbacks

@app.callback(
    [Output('logo', 'src'), Output('ticker', 'children'), Output('description', 'children')],
    [Input('submit-stock', 'n_clicks')],
    [State('stock_code', 'value')]
)
def update_data(n, stock_code):
    desc = """
    Hey! Enter stock Ticker to get information.

        1. Enter Stock ticker in the input field.
        2. Hit Submit button and wait.
        3. Click Stock Price button or Indicators button to get the stock trend.
        4. Enter the number of days (1-15) to forecast and hit the Forecast button.
        5. Hit the Yearly Analysis button for yearly insights.
    """

    # If no stock ticker is provided, return default values
    if n == 0 or stock_code == '':
        return 'https://www.linkpicture.com/q/stonks.jpg', '', desc

    try:
        # Fetch stock data using yfinance
        tk = yf.Ticker(stock_code)
        
        # Print debug information
        print(f"Attempting to fetch info for ticker: {stock_code}")
        
        # Force download of ticker info
        try:
            sinfo = tk.info
        except Exception as info_error:
            print(f"Error fetching ticker info: {info_error}")
            return 'https://www.linkpicture.com/q/stonks.jpg', 'Invalid Ticker', f'Error fetching info: {info_error}'

        # More comprehensive error checking
        if not sinfo:
            print(f"Empty info dictionary for {stock_code}")
            return 'https://www.linkpicture.com/q/stonks.jpg', 'Invalid Ticker', 'No information available for this ticker.'

        # Detailed logging of available keys
        print(f"Available keys for {stock_code}: {list(sinfo.keys())}")

        # Flexible fallback for logo and company name
        logo_url = sinfo.get('logo_url', 'https://www.linkpicture.com/q/stonks.jpg')
        short_name = sinfo.get('shortName', stock_code)
        business_summary = sinfo.get('longBusinessSummary', 'No description available.')

        # Ensure we have a valid logo URL
        if not logo_url or logo_url == '':
            logo_url = 'https://www.linkpicture.com/q/stonks.jpg'

        return logo_url, short_name, business_summary

    except Exception as e:
        # Catch-all error handling with detailed logging
        print(f"Comprehensive error for {stock_code}: {e}")
        import traceback
        traceback.print_exc()
        
        return 'https://www.linkpicture.com/q/stonks.jpg', 'Invalid Ticker', f'Error: {str(e)}'

# Update stock trend graph
@app.callback(
    Output('stonks-graph', 'children'),
    [
        Input('stock_price', 'n_clicks'),
        Input('indicators', 'n_clicks'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State('stock_code', 'value')]
)
def update_mygraph(n, ind, start, end, stock_code):
    if n == 0 or stock_code == '':
        return ''
    if start is None:
        start = date(2020, 1, 1)
    if end is None:
        end = dt.today()

    df = yf.download(stock_code, start=start, end=end)
    df.reset_index(inplace=True)
    df['ema20'] = df['Close'].rolling(20).mean()
    fig = px.line(df, x='Date', y=['Close'], title='Stock Trend')
    fig.update_traces(line_color='#ef3d3d')

    if ind % 2 != 0:
        fig.add_scatter(x=df['Date'], y=df['ema20'], line=dict(color='blue', width=1), name='EMA20')

    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Close Price")
    return dcc.Graph(figure=fig)

# Forecast graph
@app.callback(
    Output('forecast-graph', 'children'),
    [Input('Forecast', 'n_clicks'), Input('n_days', 'value')],
    [State('stock_code', 'value')]
)
def forecast(n, n_days, stock_code):
    if n == 0 or stock_code == '' or n_days == '':
        raise PreventUpdate

    try:
        n_days = int(n_days)
        if n_days < 1 or n_days > 15:
            return "Please provide a valid number of days (1-15)."

        fig = predictionModel(n_days + 1, stock_code)
        return dcc.Graph(figure=fig)
    except ValueError:
        return "Please enter a valid number for days."

# Yearly analysis graph
@app.callback(
    Output('yearly-graph', 'children'),
    [Input('yearly_analysis', 'n_clicks'), Input('n_days', 'value')],
    [State('stock_code', 'value')]
)
def yearly_analysis(n, n_days, stock_code):
    if n == 0 or stock_code == '' or n_days == '':
        raise PreventUpdate

    try:
        n_days = int(n_days)
        fig = analyze_and_visualize_forecast(stock_code, n_days)
        return fig
    except ValueError:
        return "Please enter a valid number for days."

# Yearly analysis function with pie chart
def analyze_and_visualize_forecast(stock_code, n_days):
    df = yf.download(stock_code, start="2011-01-01", end=dt.today())
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    today = dt.today()
    buy_date_str = today.strftime('%m-%d')

    results = []
    for year in range(2011, 2024):
        try:
            buy_date = dt.strptime(f"{year}-{buy_date_str}", "%Y-%m-%d")
            sell_date = buy_date + timedelta(days=n_days)
            if buy_date in df.index and sell_date in df.index:
                buy_price = df.loc[buy_date, 'Close']
                sell_price = df.loc[sell_date, 'Close']
                profit_loss = ((sell_price - buy_price) / buy_price) * 100
                results.append({'Year': year, 'Profit/Loss (%)': profit_loss})
            else:
                results.append({'Year': year, 'Profit/Loss (%)': None})
        except Exception:
            results.append({'Year': year, 'Profit/Loss (%)': None})

    results_df = pd.DataFrame(results)
    results_df['Profit/Loss (%)'].fillna(0, inplace=True)

    # Create a pie chart for the distribution of profit/loss
    positive = len(results_df[results_df['Profit/Loss (%)'] > 0])
    negative = len(results_df[results_df['Profit/Loss (%)'] < 0])
    neutral = len(results_df[results_df['Profit/Loss (%)'] == 0])

    pie_fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[positive, negative, neutral],
        hole=0.3,  # Donut chart
        marker=dict(colors=['green', 'red', 'gray'])
    )])

    pie_fig.update_layout(
        title="Profit/Loss Distribution Over the Years"
    )

    # Create the bar chart for profit/loss percentages over the years
    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            x=results_df['Year'],
            y=results_df['Profit/Loss (%)'],
            marker_color=[ 
                'green' if x > 0 else 'red' if x < 0 else 'gray' 
                for x in results_df['Profit/Loss (%)']
            ],
            name='Profit/Loss (%)'
        )
    )
    bar_fig.update_layout(
        title=f"Yearly Analysis: Buy on {buy_date_str}, Sell After {n_days} Days",
        xaxis_title="Year",
        yaxis_title="Profit/Loss (%)"
    )

    # Return both the bar chart and pie chart
    return html.Div([
        dcc.Graph(figure=bar_fig),  # Bar chart
        dcc.Graph(figure=pie_fig)   # Pie chart
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
