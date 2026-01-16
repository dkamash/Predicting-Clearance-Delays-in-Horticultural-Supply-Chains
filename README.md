# Project Title: Predictive Analytics for Optimizing "Time-to-Export" Lead Times in Kenya’s Horticultural Supply Chain
![horticulture](https://github.com/user-attachments/assets/ae3e540a-07e4-4602-b5e6-1f6a71e76238)

## 1.0 Introduction

This project develops a machine learning solution to predict the **time-to-export** (total lead time from packhouse/farm to final departure at Jomo Kenyatta International Airport — JKIA or Mombasa Port) for Kenya’s horticultural exports — primarily cut flowers, fruits, and vegetables.

The goal is to help exporters anticipate delays, reduce post-harvest losses, improve cold-chain planning, negotiate better freight contracts, and enhance overall supply chain resilience in one of Kenya’s most critical foreign exchange earning sectors.

## 2.0 Research Problem & Objectives

### Problem Statement

Kenya’s horticulture sector faces chronic and worsening logistics challenges including:

- Severe airfreight capacity shortages and skyrocketing costs at JKIA
- Frequent customs clearance delays and electronic system failures
- Port congestion and slow turnaround at Mombasa
- External disruptions (weather events, strikes, global shipping crises, ad hoc levies)

These bottlenecks result in substantial spoilage (especially for highly perishable products), missed market windows (particularly in Europe), and multi-billion-shilling revenue losses annually.

Currently, most exporters rely on experience-based estimates rather than reliable, data-driven predictions.

### Main Objective

To develop a robust, accurate machine learning regression model that predicts total export lead time (in hours or days) for Kenyan horticultural shipments, enabling proactive planning and risk mitigation.

### Specific Objectives

- Collect, clean, and deeply explore historical export logistics and external event data
- Engineer high-impact features including risk scores, seasonal indicators, and lagged delay patterns
- Build, train, and systematically compare multiple regression models (Linear Regression, KNN, XGBoost)
- Evaluate models using business-critical metrics (MAE, RMSE, R²) and custom tolerance thresholds
- Identify and rank the top drivers of export delays in the Kenyan horticulture supply chain
- Deliver a deployable model prototype and actionable recommendations for exporters, freight forwarders, and policymakers

### Stakeholders

* International freight forwarders
* Customs & port operators
* Exporters of perishable commodities
* Supply chain analytics teams

### Project Scope

* In-Scope:
  - Loading and basic exploration of the  dataset
  - Data structure & quality check (df.info(), df.describe())
  - Statistical summary of numerical features
  - Training a baseline Random Forest Classifier to predict delayed_flag
  - Automatic categorical & numerical preprocessing via Pipeline
  - Model evaluation via classification report and confusion matrix
  - Visualization of precision/recall/F1-score and confusion matrix heatmap

* Out-of-Scope :
  - Feature engineering from nested columns (documents, events)
  - Regression modeling
  - Hyperparameter tuning or cross-validation
  - Additional algorithms (XGBoost, etc.)
  - Model deployment or dashboard

## 3.0 Data Understanding

**3.1 Dataset Overview**

The dataset consists of 5,000 synthetic but realistic international consignments, stored in JSON Lines format .  
Each record represents a single shipment (mostly air freight of perishable goods such as fruits and cut flowers) and contains information about origin/destination, commodity, weight, document status, congestion, processing times, and whether the shipment was delayed beyond its SLA.

File: tlip_like_consignments_5000.jsonl  
Records: 5,000  
Columns: 27  
No missing values across the entire dataset

**3.2 Loading the Data**

The dataset is loaded into a pandas DataFrame using:


df = pd.read_json("tlip_like_consignments_5000.jsonl", lines=True)
df.head(5)

**Primary anticipated sources include:**
- KenTrade Single Window (TradeNet) — clearance times, permit records
- KNBS (Kenya National Bureau of Statistics) — monthly/quarterly export statistics
- AFA Horticultural Crops Directorate & KEPHIS — export volumes & phytosanitary certificates
- External event data — weather records, port/airport disruptions, freight cost trends


### 3.3 Variable Description

| Column                     | Type          | Description                          |
|----------------------------|---------------|--------------------------------------|
| consignment_id             | string        | Unique ID                            |
| created_at                 | datetime      | Creation timestamp                   |
| origin_country             | string        | Origin country                       |
| destination_country        | string        | Destination country                  |
| origin_port                | string        | Origin port                          |
| destination_port           | string        | Destination port                     |
| shipment_mode              | string        | AIR or SEA                           |
| commodity                  | string        | Product type                         |
| hs_code                    | integer       | HS code                              |
| gross_weight_kg            | float         | Weight (kg)                          |
| declared_value_usd         | float         | Value (USD)                          |
| doc_completeness_score     | float         | Doc readiness (0-1)                  |
| missing_docs_proxy         | integer       | Missing docs count                   |
| doc_amendments             | integer       | Doc amendments                       |
| congestion_index           | float         | Congestion level (0-1)               |
| is_weekend_created         | integer       | Weekend flag                         |
| customs_release_hours      | float         | Customs time (h)                     |
| terminal_dwell_hours       | float         | Terminal time (h)                    |
| sla_hours                  | integer       | SLA target (h)                       |
| total_processing_hours     | float         | Total time (h)                       |
| delayed_flag               | integer       | Delayed? (1=yes)                     |
| delay_hours                | float         | Delay hours                          |
| documents                  | list of dicts | Documents info                       |
| events                     | list of dicts | Events timeline                      |

## 4.0 Data Cleaning & Pre-processing

#### 4.1 Handling Missing Values
- The notebook runs `df.info()`
- Result: All 27 columns have 5000 non-null values → **no missing data**
- No imputation, dropping, or filling of missing values is performed (none needed)


