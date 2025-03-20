import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



##dp
def validate_csv_columns(df):
    required_columns = ['Date', 'Day', 'Product Name', 'Quantity', 'Price']
    return all(column in df.columns for column in required_columns)

def process_data(df):

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['Quantity'] = pd.to_numeric(df['Quantity'])
        df['Price'] = pd.to_numeric(df['Price'])

        df['Total Sales'] = df['Price']
        # df['Total Sales'] = df['Quantity'] * df['Price']
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        return df, None
    except Exception as e:
        return None, str(e)

def get_summary_stats(df):

    stats = {
        'total_sales': df['Total Sales'].sum(),
        'total_products': df['Product Name'].nunique(),
        'avg_daily_sales': df.groupby('Date')['Total Sales'].sum().mean(),
        'best_selling_product': df.groupby('Product Name')['Quantity'].sum().idxmax(),
        'best_day': df.groupby('Day')['Total Sales'].sum().idxmax()
    }
    return stats




##dv
def create_daily_sales_trend(df):
 
    daily_sales = df.groupby('Date')['Total Sales'].sum().reset_index()
    fig = px.line(daily_sales, x='Date', y='Total Sales',
                  title='Daily Sales Overview',
                  template='plotly_white')
    fig.update_traces(line_color='#2E86C1', line_width=2)
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Total Sales ($)',
        hovermode='x unified'
    )
    return fig

def create_detailed_daily_trend(df):
   

    daily_product_sales = df.groupby(['Date', 'Product Name'])['Total Sales'].sum().reset_index()

    top_products = df.groupby('Product Name')['Total Sales'].sum().nlargest(5).index
    filtered_data = daily_product_sales[daily_product_sales['Product Name'].isin(top_products)]

    fig = px.line(filtered_data, x='Date', y='Total Sales', 
                  color='Product Name',
                  title='Detailed Daily Sales by Top 5 Products',
                  template='plotly_white')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        legend_title='Products',
        hovermode='x unified'
    )
    return fig

def create_product_sales_chart(df):
 
    product_sales = df.groupby('Product Name')['Total Sales'].sum().sort_values(ascending=True).tail(10)
    fig = px.bar(product_sales,
                 title='Top 10 Products by Sales',
                 template='plotly_white',
                 orientation='h')
    fig.update_traces(marker_color='#27AE60')
    fig.update_layout(
        xaxis_title='Total Sales ($)',
        yaxis_title='Product Name',
        showlegend=False
    )
    return fig

def create_detailed_product_analysis(df):
   
    product_analysis = df.groupby('Product Name').agg({
        'Quantity': 'sum',
        'Total Sales': 'sum'
    }).reset_index().nlargest(15, 'Total Sales')

    fig = px.scatter(product_analysis, 
                     x='Quantity', 
                     y='Total Sales',
                     text='Product Name',
                     title='Top 15 Products Performance Analysis',
                     template='plotly_white')

    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title='Total Quantity Sold',
        yaxis_title='Total Revenue ($)'
    )
    return fig

def create_weekly_pattern(df):
   
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = df.groupby('Day')['Total Sales'].mean().reindex(day_order).reset_index()

    fig = px.bar(weekly_avg, x='Day', y='Total Sales',
                 title='Average Daily Sales',
                 template='plotly_white')
    fig.update_traces(marker_color='#8E44AD')
    fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Average Sales ($)',
        xaxis={'categoryorder': 'array', 'categoryarray': day_order}
    )
    return fig

def create_product_heatmap(df):
  

    top_products = df.groupby('Product Name')['Total Sales'].sum().nlargest(10).index
    filtered_df = df[df['Product Name'].isin(top_products)]

    pivot_table = filtered_df.pivot_table(
        values='Quantity',
        index='Product Name',
        columns='Day',
        aggfunc='sum',
        fill_value=0
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlBu'))

    fig.update_layout(
        title='Top 10 Products Sales Heatmap by Day',
        template='plotly_white',
        height=400
    )
    return fig

def create_sales_summary(df):
   
    monthly_sales = df.groupby(['Year', 'Month'])['Total Sales'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))

    fig = px.line(monthly_sales, x='Date', y='Total Sales',
                  title='Monthly Sales Trend',
                  template='plotly_white')
    fig.update_traces(line_color='#E67E22', line_width=2)
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Sales ($)',
        hovermode='x unified'
    )
    return fig



##p
def prepare_prediction_data(df):

    df_prep = df.copy()

    df_prep['DayOfWeek'] = pd.Categorical(df_prep['Day']).codes
    df_prep['ProductCode'] = pd.Categorical(df_prep['Product Name']).codes
    df_prep['Month'] = df_prep['Date'].dt.month

    return df_prep

def train_prediction_model(df):
  

    df_prep = prepare_prediction_data(df)

    X = df_prep[['DayOfWeek', 'ProductCode', 'Month']]
    y = df_prep['Quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def get_inventory_recommendations(df, model):
  
    df_prep = prepare_prediction_data(df)

    recommendations = []

    for product in df['Product Name'].unique():
        product_data = df_prep[df_prep['Product Name'] == product]
        avg_sales = product_data['Quantity'].mean()

        next_week_pred = []
        for day in range(7):
            X_pred = pd.DataFrame({
                'DayOfWeek': [day],
                'ProductCode': [product_data['ProductCode'].iloc[0]],
                'Month': [df_prep['Month'].iloc[-1]]
            })
            pred = model.predict(X_pred)[0]
            next_week_pred.append(pred)

        avg_predicted = np.mean(next_week_pred)

        if avg_predicted > avg_sales * 1.2:
            recommendations.append(f"📈 Increase {product} inventory by {int((avg_predicted/avg_sales - 1) * 100)}%")
        elif avg_predicted < avg_sales * 0.8:
            recommendations.append(f"📉 Decrease {product} inventory by {int((1 - avg_predicted/avg_sales) * 100)}%")
        else:
            recommendations.append(f"✅ Maintain current inventory levels for {product}")

    return recommendations






st.set_page_config(
    page_title="Get Sales Insights & Recommendations",
    layout="wide"
)

page = st.sidebar.radio("Navigate to", ["Sales Analysis", "Sales Predictions"])

st.title("Get Sales Insights & Recommendations")
st.markdown("""
Upload your sales data CSV file to get insights and predictions.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if not validate_csv_columns(df):
        st.error("Invalid CSV format. Please ensure your CSV has the required columns.")
    else:
      
        processed_df, error = process_data(df)

        if error:
            st.error(f"Error processing data: {error}")
        else:
            if page == "Sales Analysis":
                
                stats = get_summary_stats(processed_df)

                st.markdown("Key Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sales", f"₹{stats['total_sales']:,.2f}")
                with col2:
                    st.metric("Total Products", stats['total_products'])
                with col3:
                    st.metric("Avg Daily Sales", f"₹{stats['avg_daily_sales']:,.2f}")

                st.markdown("Sales Analysis")

                tab1, tab2 = st.tabs(["Quick Overview", "Detailed Analysis"])

                with tab1:
                    st.plotly_chart(create_daily_sales_trend(processed_df), use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_product_sales_chart(processed_df), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_weekly_pattern(processed_df), use_container_width=True)

                    st.plotly_chart(create_sales_summary(processed_df), use_container_width=True)

                with tab2:
                    st.plotly_chart(create_detailed_daily_trend(processed_df), use_container_width=True)
                    st.plotly_chart(create_detailed_product_analysis(processed_df), use_container_width=True)
                    st.plotly_chart(create_product_heatmap(processed_df), use_container_width=True)

                st.markdown("Download Processed Data")
                csv = processed_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="processed_sales_data.csv",
                    mime="text/csv"
                )

            else:
                st.markdown("**Sales Predictions & Recommendations**")
                st.markdown("""
                Our prediction model analyzes your historical sales data to provide inventory recommendations.
                Here's how it works:

                1. **Historical Analysis**: We analyze past sales patterns across different days and products
                2. **Pattern Recognition**: The model identifies trends in daily and weekly sales
                3. **Future Projection**: Based on these patterns, we predict future sales volumes
                """)

                with st.spinner("Generating predictions..."):
                    model, X_test, y_test = train_prediction_model(processed_df)
                    recommendations = get_inventory_recommendations(processed_df, model)

                st.markdown("**Inventory Recommendations**")
                st.markdown("""
                The recommendations below are based on:
                - Historical sales patterns
                - Day-of-week trends
                - Seasonal variations
                - Product-specific performance
                """)

                for rec in recommendations:
                    st.info(rec)

                st.markdown("""
                ### Understanding the Recommendations

                - 📈 **Increase Inventory**: Suggests higher future demand based on positive trends
                - 📉 **Decrease Inventory**: Indicates potential overstock based on declining demand
                - ✅ **Maintain Levels**: Current inventory levels align with predicted demand

                """)

else:
    st.info("""
    ### Your CSV file should have the columns:

    Date, Day, Product Name, Quantity, Price
    """)
