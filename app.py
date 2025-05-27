import os
from flask import Flask

from controller.honey_controller import honey_bp
from controller.predict_controller import predict_bp
from controller.upload_controller import upload_bp

app = Flask(__name__)

# ─── configure your UPLOAD_FOLDER ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# this must point at the folder that contains your 'uploads' sub-folder
UPLOAD_FOLDER = os.path.join(BASE_DIR,  'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ──────────────────────────────────────────────────────────────────────────────

# Image-based bee health prediction
app.register_blueprint(predict_bp, url_prefix="/bee-health")

# Honey quality prediction
app.register_blueprint(honey_bp,   url_prefix="/honey-quality")

# Upload / serve images:
# your upload_controller should have a route like:
#    @upload_bp.route('/uploads/<filename>')
# so the full URL becomes `/upload/uploads/<filename>`
app.register_blueprint(upload_bp, url_prefix='/upload')

if __name__ == "__main__":
    app.run(debug=True)
