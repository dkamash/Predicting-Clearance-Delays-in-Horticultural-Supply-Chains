# Predicting-Clearance-Delays-in-Horticultural-Supply-Chains
A predictive model that estimates the risk of clearance delays for horticultural export consignments from East Africa to Europe using synthetic trade data. 

## Local deployment (Streamlit)

1. Train and pickle the baseline logistic regression model:

```
python train_logreg_model.py --data-path tlip_like_consignments_5000.jsonl --model-path models/log_reg_pipeline.pkl
```

2. Launch the Streamlit app:

```
streamlit run app.py
```
