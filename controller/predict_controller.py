from flask import Blueprint, request, jsonify
from service.predict_service import predict_bee_health

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        prediction = predict_bee_health(file)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
