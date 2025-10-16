#!/bin/bash
python rtr_and_ortools.py --orders /app/data/test/ml_ozon_logistic_dataSetOrders.json --couriers /app/data/test/ml_ozon_logistic_dataSetCouriers.json --durations_json /app/data/test/ml_ozon_logistic_dataDurations.json --durations_db /app/data/test/durations.sqlite --output /app/data/solution.json
