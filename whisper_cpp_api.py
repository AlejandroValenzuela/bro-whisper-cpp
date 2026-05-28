#!/usr/bin/env python3
from flask import Flask, request, jsonify
import subprocess
import json
import os
import logging
from pathlib import Path
import time

app = Flask(__name__)

WHISPER_CPP_PATH = "/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/whisper.cpp/models/ggml-tiny.bin"
TEMP_DIR = "/app/temp_audio"
MAX_FILE_SIZE = 25 * 1024 * 1024

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "service": "whisper-cpp", "model": "tiny", "version": "1.0"}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    start_time = time.time()
    tmp_path = None

    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided", "status": "error"}), 400

        audio_file = request.files['audio']

        # SIEMPRE usar .wav (whisper.cpp lo acepta mejor)
        tmp_filename = f"audio_{int(time.time() * 1000)}.wav"
        tmp_path = os.path.join(TEMP_DIR, tmp_filename)
        audio_file.save(tmp_path)

        file_size = os.path.getsize(tmp_path)
        logger.info(f"Saved audio: {tmp_path} ({file_size} bytes)")

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
            return jsonify({"error": "Transcription failed", "details": result.stderr[:200], "status": "error"}), 500

        try:
            output = json.loads(result.stdout)
            transcription = output.get('result', [{}])[0].get('text', '').strip()

            if not transcription:
                return jsonify({"error": "No speech detected", "status": "error"}), 400

            processing_time = time.time() - start_time
            return jsonify({
                "text": transcription,
                "language": language,
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
    return jsonify({"service": "Whisper.CPP API", "version": "1.0", "model": "tiny"}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "status": "error"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
