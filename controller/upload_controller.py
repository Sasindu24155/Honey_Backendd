from flask import Blueprint, request, jsonify, send_from_directory, current_app
from service.upload_service import handle_image_upload
import os

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        saved_filename = handle_image_upload(file)
        return jsonify({
            'message': 'Upload successful',
            'filename': saved_filename,
            'download_url': f"/upload/uploads/{saved_filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🆕 Serve uploaded images
@upload_bp.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    upload_folder = current_app.config['UPLOAD_FOLDER']
    return send_from_directory(upload_folder, filename)
