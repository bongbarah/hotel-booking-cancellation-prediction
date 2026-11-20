# Hotel Booking Cancellation & Revenue Intelligence

Portfolio project using the Hotel Booking Demand dataset.

Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand  
Research DOI: https://doi.org/10.1016/j.dib.2018.11.126

1. Download `hotel_bookings.csv` and put it in `data/raw/`.
2. Install packages: `pip install -r requirements.txt`
3. Run: `python run_pipeline.py`
4. Explain model: `python -c "from src.explain import explain_model; print(explain_model())"`
5. Launch: `streamlit run app.py`

The model excludes reservation status/date and variables not safely available at booking time. Exact duplicate rows are removed; conflicting predictor profiles with different target labels are retained and grouped into the same train/test partition. The main model selection metric is PR-AUC.

Revenue exposure is estimated as ADR multiplied by nights. It is not realised revenue or a confirmed loss. Data spans 2015–2017, so current deployment would require recent validation and calibration.
