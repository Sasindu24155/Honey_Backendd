from flask import Blueprint, request, jsonify
from service.honey_service import predict_honey_quality

honey_bp = Blueprint("honey", __name__)

@honey_bp.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        results = predict_honey_quality(data)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
