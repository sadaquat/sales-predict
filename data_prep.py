import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_monthly_revenue():
    """Extract monthly revenue from Northwind using SemanticSQL's revenue definition."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    # Same revenue definition as SemanticSQL semantic layer
    # net of discounts, shipped orders only
    query = """
        SELECT
            DATE_TRUNC('month', o.order_date) AS month,
            SUM(od.unit_price * od.quantity * (1 - od.discount)) AS revenue,
            COUNT(DISTINCT o.order_id) AS order_count,
            COUNT(DISTINCT o.customer_id) AS unique_customers,
            AVG(od.unit_price * od.quantity * (1 - od.discount)) AS avg_order_value
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        WHERE o.shipped_date IS NOT NULL
        GROUP BY DATE_TRUNC('month', o.order_date)
        ORDER BY month
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['month'] = pd.to_datetime(df['month'])
    df = df.rename(columns={'month': 'ds', 'revenue': 'y'})

    print(f"Loaded {len(df)} months of data")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Total revenue: ${df['y'].sum():,.2f}")
    print(f"\nSample data:")
    print(df.head())

    avg_revenue = df['y'].median()
    df = df[df['y'] > avg_revenue * 0.5].copy()
    df = df.reset_index(drop=True)

    print(f"After removing partial months: {len(df)} months")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    return df


def get_revenue_by_category():
    """Revenue breakdown by product category per month."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    query = """
        SELECT
            DATE_TRUNC('month', o.order_date) AS month,
            c.category_name,
            SUM(od.unit_price * od.quantity * (1 - od.discount)) AS revenue
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        WHERE o.shipped_date IS NOT NULL
        GROUP BY DATE_TRUNC('month', o.order_date), c.category_name
        ORDER BY month, revenue DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()
    df['month'] = pd.to_datetime(df['month'])
    
    # Remove partial last month (same as main revenue query)
    avg_by_month = df.groupby('month')['revenue'].sum()
    avg_rev = avg_by_month.median()
    valid_months = avg_by_month[avg_by_month > avg_rev * 0.5].index
    df = df[df['month'].isin(valid_months)].copy()
    return df


def get_revenue_by_country():
    """Revenue by customer country per month."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    query = """
        SELECT
            DATE_TRUNC('month', o.order_date) AS month,
            c.country,
            SUM(od.unit_price * od.quantity * (1 - od.discount)) AS revenue
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.shipped_date IS NOT NULL
        GROUP BY DATE_TRUNC('month', o.order_date), c.country
        ORDER BY month, revenue DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()
    df['month'] = pd.to_datetime(df['month'])
    
    # Remove partial last month (same as main revenue query)
    avg_by_month = df.groupby('month')['revenue'].sum()
    avg_rev = avg_by_month.median()
    valid_months = avg_by_month[avg_by_month > avg_rev * 0.5].index
    df = df[df['month'].isin(valid_months)].copy()
    return df


if __name__ == "__main__":
    df = get_monthly_revenue()
    df.to_csv("data/monthly_revenue.csv", index=False)
    print("\nSaved to data/monthly_revenue.csv")