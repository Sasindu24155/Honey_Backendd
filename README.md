# Bee Honey Backend 

Backend API for:
- Bee health detection (image)
- Honey quality prediction (CSV)

## Setup

1. **Clone repo & go to folder**
```bash
git clone <your-repo-link>
cd Backend

    2.    Create virtual environment & activate

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

    3.    Install dependencies

pip install flask torch pandas scikit-learn numpy Pillow

    4.    Run app

python app.py

Runs on: http://127.0.0.1:5000/

Datasets

    •    Bee Image Dataset [https://www.kaggle.com/datasets/raghav723/predict-price-of-honey](https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images)
    •    Honey Quality Dataset https://www.kaggle.com/datasets/raghav723/predict-price-of-honey

Endpoints

    •    POST /bee-health/predict — Image upload
    •    POST /honey-quality/predict — CSV/feature input
    •    GET /upload/uploads/<filename> — Access uploaded files


Place datasets & models in the project root or upload/uploads/.

## Folder Structure

Backend/
│
├── controller/
│   ├── honey_controller.py
│   ├── predict_controller.py
│   └── upload_controller.py
│
├── service/
│   ├── honey_service.py
│   ├── predict_service.py
│   └── upload_service.py
│
├── upload/
│   └── uploads/  # Folder to store uploaded files
│
├── app.py        # Main application entry point
├── bee_data.csv
├── honey_model.pkl
├── bee_health_classifier.pth
├── quality_test.py
├── test.py
├── training.py
├── README.md     

---