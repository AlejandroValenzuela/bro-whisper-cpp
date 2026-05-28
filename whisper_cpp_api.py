#!/usr/bin/env python3
from flask import Flask, request, jsonify
import subprocess
import json
import os
import logging
from pathlib import Path
import time

app = Flask(__name__)

WHISPER_CPP_PATH ="/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/whisper.cpp/models/ggml-tiny.bin"
TEMP_DIR = "/app/temp_audio"
MAX_FILE_SIZE = 25 * 1024 * 1024
ALLOWED_FORMATS = {'wav', 'mp3', 'ogg', 'm4a'}

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify({
            "status": "running",
            "service": "whisper-cpp",
            "model": "tiny",
            "version": "1.0"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    start_time = time.time()
    tmp_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided", "status": "error"}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({"error": "No selected file", "status": "error"}), 400
        
        file_ext = audio_file.filename.split('.')[-1].lower()
        if file_ext not in ALLOWED_FORMATS:
            return jsonify({"error": f"Format not allowed", "status": "error"}), 400
        
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "File too large", "status": "error"}), 413
        
        tmp_filename = f"audio_{int(time.time() * 1000)}.{file_ext}"
        tmp_path = os.path.join(TEMP_DIR, tmp_filename)
        audio_file.save(tmp_path)
        logger.info(f"Saved audio: {tmp_path}")
        
        language = request.form.get('language', 'es')
        
        cmd = [
            WHISPER_CPP_PATH,
            "-m", MODEL_PATH,
            "-l", language,
            "-f", tmp_path,
            "-of", "json",
            "-t", str(os.cpu_count() or 4),
            "--no-prints"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"Whisper error: {result.stderr}")
            return jsonify({"error": "Transcription failed", "status": "error"}), 500
        
        try:
            output = json.loads(result.stdout)
            transcription = output.get('result', [{}])[0].get('text', '').strip()
            
            if not transcription:
                return jsonify({"error": "No speech detected", "status": "error"}), 400
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "text": transcription,
                "language": language,
                "confidence": 0.95,
                "processing_time": round(processing_time, 2),
                "status": "success"
            }), 200
        
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid response format", "status": "error"}), 500
    
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout", "status": "error"}), 504
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "service": "Whisper.CPP API",
        "version": "1.0",
        "model": "tiny",
        "languages": ["es", "en", "fr", "de", "it", "pt"],
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "allowed_formats": list(ALLOWED_FORMATS),
        "endpoints": {
            "/health": "GET - Health check",
            "/transcribe": "POST - Transcribe audio",
            "/info": "GET - Service info"
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status": "error"}), 500

if __name__ == '__main__':
    logger.info("Starting Whisper.CPP API on port 5002...")
    app.run(host='0.0.0.0', port=5002)
