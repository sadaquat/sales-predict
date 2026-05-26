##### \# SalesPredict — ML Revenue Forecasting

##### 

##### > Classical machine learning applied to enterprise sales data — predicting future revenue trends using Facebook Prophet.

##### 

##### \*\*🔗 Live demo:\*\* \[sales-predict-sem-sql.streamlit.app](https://sales-predict-sem-sql.streamlit.app)

##### 

##### \---

##### 

##### \## What this project demonstrates

##### 

##### Most AI/ML portfolio projects only show LLM API calls. SalesPredict demonstrates \*\*classical ML fundamentals\*\*:

##### 

##### \- Extracting and preparing real data from a PostgreSQL database

##### \- Training a forecasting model with proper \*\*train/test split\*\* (19 months train, 3 months held out)

##### \- Evaluating model accuracy using \*\*MAE, RMSE, and MAPE\*\*

##### \- Deploying a production ML pipeline on cloud infrastructure

##### 

##### \---

##### 

##### \## The dataset

##### 

##### Uses the \*\*Northwind sample database\*\* — the same dataset as \[SemanticSQL](https://genai-reporting-agent.streamlit.app).

##### 

##### Revenue is calculated using the same semantic definition across both projects:

##### 

##### ```sql

##### SUM(od.unit\_price \* od.quantity \* (1 - od.discount))

##### WHERE o.shipped\_date IS NOT NULL

##### ```

##### 

##### This is intentional — both projects use consistent business logic. In a real enterprise, this definition would live in a semantic layer (which SemanticSQL demonstrates).

##### 

##### \---

##### 

##### \## Model results

##### 

##### | Metric | Value | Meaning |

##### |--------|-------|---------|

##### | \*\*MAE\*\* | $38,631 | Average error per month |

##### | \*\*RMSE\*\* | $48,933 | Penalises large errors more than MAE |

##### | \*\*MAPE\*\* | 38.1% | Percentage error — expected for 22 months of data |

##### 

##### MAPE of 38.1% is expected with only 22 months of training data. Prophet typically achieves under 15% MAPE with 5+ years of historical data.

##### 

##### \---

##### 

##### \## Architecture

##### 

##### PostgreSQL (Northwind / Neon cloud)

##### ↓

##### data\_prep.py — SQL queries extract monthly revenue, category, country breakdowns

##### ↓

##### model.py — Prophet trains on 19 months, evaluates on 3 months holdout

##### ↓

##### app.py — Streamlit dashboard with Plotly visualizations

##### ↓

##### Streamlit Cloud — live deployment

##### 

##### \---

##### 

##### \## Dashboard features

##### 

##### \- \*\*KPI cards\*\* — total revenue, average monthly, 6-month forecast, model accuracy

##### \- \*\*Forecast chart\*\* — historical data + predicted future + 95% confidence interval

##### \- \*\*Accuracy panel\*\* — actual vs predicted bar chart + MAE/RMSE/MAPE explained

##### \- \*\*Category breakdown\*\* — revenue by product category (bar chart + trend lines)

##### \- \*\*Country breakdown\*\* — top 15 countries by total revenue

##### \- \*\*Adjustable forecast horizon\*\* — 3 to 12 months via sidebar slider

##### \- \*\*Trend components toggle\*\* — shows underlying trend and seasonality

##### 

##### \---

##### 

##### \## Stack

##### 

##### \- \*\*Forecasting:\*\* Facebook Prophet

##### \- \*\*Data processing:\*\* pandas, NumPy, scikit-learn

##### \- \*\*Database:\*\* PostgreSQL (Neon cloud)

##### \- \*\*Visualization:\*\* Plotly

##### \- \*\*UI:\*\* Streamlit

##### \- \*\*Hosting:\*\* Streamlit Cloud

##### 

##### \---

##### 

##### \## Running locally

##### 

##### ```bash

##### \# 1. Clone the repo

##### git clone https://github.com/sadaquat/sales-predict.git

##### cd sales-predict

##### 

##### \# 2. Start PostgreSQL with Northwind data

##### docker-compose up -d  # OR use your own PostgreSQL instance

##### 

##### \# 3. Set up Python environment

##### python -m venv venv

##### .\\venv\\Scripts\\Activate.ps1   # Windows

##### source venv/bin/activate       # Mac/Linux

##### 

##### \# 4. Install dependencies

##### pip install -r requirements.txt

##### 

##### \# 5. Set environment variables

##### \# Create .env file with:

##### \# DATABASE\_URL=postgresql://postgres:postgres@localhost:5432/northwind

##### 

##### \# 6. Run the app

##### streamlit run app.py

##### ```

##### 

##### \---

##### 

##### \## Part of a portfolio

##### 

##### This project is part of a two-project portfolio demonstrating the full ML/AI stack:

##### 

##### | Project | Focus | Live demo |

##### |---------|-------|-----------|

##### | \*\*SemanticSQL\*\* | GenAI · RAG · LLM · Semantic layer | \[genai-reporting-agent.streamlit.app](https://genai-reporting-agent.streamlit.app) |

##### | \*\*SalesPredict\*\* | Classical ML · Forecasting · Evaluation | \[sales-predict-sem-sql.streamlit.app](https://sales-predict-sem-sql.streamlit.app) |

##### 

##### Both projects use the same Northwind dataset with the same semantic revenue definition — demonstrating portfolio coherence across GenAI and classical ML.

##### 

##### \---

##### 

##### \## About

##### 

##### Built by \*\*Sadaquat Khan\*\* — Senior BI Consultant transitioning into AI/ML Engineering.

##### 

##### \[LinkedIn](https://linkedin.com/in/sadaquat-khan) · \[GitHub](https://github.com/sadaquat/semantic-sql) · \[SemanticSQL](https://genai-reporting-agent.streamlit.app)

