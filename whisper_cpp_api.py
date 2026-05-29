from flask import Flask, request, jsonify
import subprocess, os, logging, time
from pathlib import Path

app = Flask(__name__)
WHISPER_CPP_PATH = "/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/whisper.cpp/models/ggml-tiny.bin"
TEMP_DIR = "/app/temp_audio"
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
        tmp_path = os.path.join(TEMP_DIR, f"audio_{int(time.time() * 1000)}.wav")
        audio_file.save(tmp_path)
        logger.info(f"Saved audio: {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

        language = request.form.get('language', 'es')
        cmd = [WHISPER_CPP_PATH, "-m", MODEL_PATH, "-l", language, "-f", tmp_path, "-nt", "-t", str(os.cpu_count() or 4)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        output = result.stdout + result.stderr
        logger.info(f"Whisper output: {output[:200]}")

        import re
        match = re.match(r'^(.*?)whisper_', output, re.DOTALL)
        raw = match.group(1) if match else ""
        lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.strip().startswith("[")]
        transcription = " ".join(lines).strip()
        if not transcription:
            transcription = "..."

        processing_time = time.time() - start_time
        logger.info(f"Transcription: '{transcription}' ({processing_time:.2f}s)")

        return jsonify({"text": transcription, "language": language, "processing_time": round(processing_time, 2), "status": "success"}), 200

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout", "status": "error"}), 504
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

@app.route('/info', methods=['GET'])
def info():
    return jsonify({"service": "Whisper.CPP API", "version": "1.0", "model": "tiny"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)