"""
MultiLLM Cost-Optimized API Microservice
"""
import os
import json
import time
import yaml
from flask import Flask, request, jsonify , render_template
from services.provider_manager import ProviderManager
from utils.logger import setup_logger

# Initialize Flask app
app = Flask(__name__)

# Setup logger
logger = setup_logger()

# Load provider configuration
def load_config():
    config_path = os.environ.get('CONFIG_PATH', 'config/providers.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

# Initialize provider manager
provider_manager = None

@app.before_request
def initialize():
    global provider_manager
    config = load_config()
    provider_manager = ProviderManager(config)
    logger.info("Provider manager initialized with configuration.")
@app.route('/')
def home():
   return render_template('index.html')
@app.route('/generate', methods=['POST'])

@app.route('/generate', methods=['POST'])
def generate():
    """
    Endpoint to generate text using the most cost-effective LLM provider.

    Accepts both application/json and form-data.

    Request formats:
    - JSON:
      {
        "prompt": "Hello!",
        "max_tokens": 100,
        "temperature": 0.7
      }

    - Form:
      prompt=Hello!&max_tokens=100&temperature=0.7
    """

    start_time = time.time()

    # Detect if JSON or form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    if not data or 'prompt' not in data:
        return jsonify({
            "error": "Missing required parameter: prompt"
        }), 400

    # Convert types as needed
    prompt = data['prompt']
    max_tokens = int(data.get('max_tokens', 100))
    temperature = float(data.get('temperature', 0.7))

    try:
        # Call your LLM provider manager
        result = provider_manager.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        time_taken = time.time() - start_time
        result['timeTaken'] = round(time_taken, 2)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({
            "error": "Failed to generate response",
            "details": str(e),
            "timeTaken": round(time.time() - start_time, 2)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get usage statistics and logs."""
    try:
        # Read usage logs from storage
        with open('storage/usage_logs.json', 'r') as file:
            logs = json.load(file)
        
        # Calculate summary statistics
        summary = {
            "totalRequests": len(logs),
            "totalCost": sum(log.get('cost', 0) for log in logs),
            "totalTokens": sum(log.get('tokens', {}).get('total', 0) for log in logs),
            "providerUsage": {}
        }
        
        # Count usage by provider
        for log in logs:
            provider = log.get('modelUsed')
            if provider:
                if provider not in summary['providerUsage']:
                    summary['providerUsage'][provider] = 0
                summary['providerUsage'][provider] += 1
        
        return jsonify({
            "summary": summary,
            "recentLogs": logs[-50:]  # Return the most recent 50 logs
        })
    
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve statistics",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "providers": provider_manager.get_provider_status() if provider_manager else []
    })

if __name__ == '__main__':
    # Ensure storage directory exists
    os.makedirs('storage', exist_ok=True)
    
    # Create empty usage logs file if it doesn't exist
    if not os.path.exists('storage/usage_logs.json'):
        with open('storage/usage_logs.json', 'w') as file:
            json.dump([], file)
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)