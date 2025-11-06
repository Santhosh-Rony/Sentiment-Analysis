# 🤖 Sentiment Analysis with DistilBERT

A production-ready sentiment analysis API using pre-trained DistilBERT from Hugging Face Transformers. Features a modern web interface, RESTful API, and FastAPI backend.

## 📚 Tech Stack

- **Pre-trained Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **ML Framework**: PyTorch (for Hugging Face transformers)
- **API Framework**: FastAPI
- **Frontend**: Modern HTML/CSS/JavaScript
- **Available for Custom Development**: TensorFlow + Keras

> **Note**: This project uses PyTorch for the transformers library because most Hugging Face pre-trained models are in PyTorch format. TensorFlow and Keras are also installed and available for your own custom model development.

## 🌟 Features

- ✅ Real-time sentiment analysis (Positive/Negative)
- ✅ Confidence scores and probability distributions
- ✅ Single text and batch processing
- ✅ RESTful API with automatic documentation
- ✅ Modern, responsive web interface
- ✅ Docker support for easy deployment
- ✅ Health check and monitoring endpoints

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "DL Project"
```

2. **Create and activate virtual environment**
```bash
# Create new environment
python -m venv tf_env

# Activate on macOS/Linux
source tf_env/bin/activate

# Activate on Windows
tf_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for transformers pre-trained models)
- TensorFlow + Keras (for custom model development)
- FastAPI + Uvicorn (for the API server)
- Transformers (Hugging Face library)

4. **Start the server**
```bash
python main.py --port 8000
```

5. **Access the application**
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

## 📂 Project Structure

```
DL Project/
│
├── api/
│   ├── app.py              # FastAPI application with endpoints
│   └── static/
│       └── index.html      # Modern web UI
│
├── main.py                 # Application entry point
├── use_pretrained_model.py # Standalone sentiment analysis script
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── example_requests.http   # API testing examples
```

## 🔧 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "backend": "pytorch",
  "note": "PyTorch for transformers, TensorFlow available for custom models"
}
```

### Single Text Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This movie is absolutely fantastic!",
    "return_probabilities": true
  }'
```

**Response:**
```json
{
  "text": "This movie is absolutely fantastic!",
  "sentiment": "Positive",
  "confidence": 0.9998,
  "probabilities": {
    "positive": 0.9998,
    "negative": 0.0002
  }
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product!",
      "Terrible service.",
      "It was okay."
    ],
    "return_probabilities": true
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Great product!",
      "sentiment": "Positive",
      "confidence": 0.9995,
      "probabilities": {
        "positive": 0.9995,
        "negative": 0.0005
      }
    },
    ...
  ],
  "total_processed": 3
}
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t sentiment-analysis .

# Run the container
docker run -p 8000:8000 sentiment-analysis

# Access at http://localhost:8000
```

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  sentiment-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
```

Run with:
```bash
docker-compose up
```

## 🛠 Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --host TEXT         Host address (default: 0.0.0.0)
  --port INTEGER      Port number (default: 8000)
  --reload           Enable auto-reload for development
  --log-level TEXT    Log level: debug, info, warning, error (default: info)
```

## 📊 Model Information

**Model**: `distilbert-base-uncased-finetuned-sst-2-english`

- **Type**: DistilBERT (distilled BERT)
- **Task**: Binary sentiment classification
- **Classes**: Positive, Negative
- **Source**: Hugging Face Model Hub
- **Training**: Fine-tuned on SST-2 (Stanford Sentiment Treebank)
- **Performance**: ~94% accuracy on validation set

### Why DistilBERT?

- ✅ 40% smaller than BERT
- ✅ 60% faster inference
- ✅ Retains 97% of BERT's performance
- ✅ Perfect for production deployment

## 💡 Using TensorFlow + Keras

While this project uses PyTorch for the pre-trained transformers model, TensorFlow and Keras are installed and ready for your custom model development.

### Example: Custom TensorFlow Model

```python
import tensorflow as tf
from tensorflow import keras

# Create a simple sentiment model
model = keras.Sequential([
    keras.layers.Embedding(10000, 128),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train your model
# model.fit(train_data, train_labels, epochs=10)
```

You can integrate your custom TensorFlow/Keras models into the FastAPI application by adding new endpoints in `api/app.py`.

## 🔍 API Documentation

When the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve HTML web interface |
| GET | `/health` | Health check and model status |
| POST | `/predict` | Single text sentiment analysis |
| POST | `/predict/batch` | Batch text sentiment analysis |

## 🧪 Testing

### Using the Web Interface

1. Open `http://localhost:8000` in your browser
2. Enter text in the input box
3. Click "ANALYZE SENTIMENT" or press Ctrl/Cmd + Enter
4. View results with confidence scores

### Using the Standalone Script

```bash
python use_pretrained_model.py
```

This will run test cases and show sentiment predictions.

### Using the HTTP Examples

Open `example_requests.http` in VS Code with the REST Client extension, or use the curl commands provided in the file.

## 📈 Performance Considerations

- **Model Loading**: Model loads once at startup (~2-3 seconds)
- **Inference Speed**: ~50-100ms per text on CPU
- **Batch Processing**: More efficient for multiple texts
- **GPU Support**: Automatically uses GPU if available (CUDA/MPS)
- **Memory Usage**: ~500MB for model in memory

## 🔧 Troubleshooting

### Server Won't Start

```bash
# Check if port is already in use
lsof -i :8000

# Try a different port
python main.py --port 8001
```

### Module Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Slow Predictions

- Use batch prediction for multiple texts
- Consider GPU acceleration (install CUDA)
- Reduce max_length parameter if texts are short

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments

Built with:
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained models
- [PyTorch](https://pytorch.org/) - Deep learning framework for transformers
- [TensorFlow](https://www.tensorflow.org/) - Available for custom models
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) - Efficient transformer model

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the troubleshooting section above

---

1. Framework Used: PyTorch + TensorFlow (Both!)
Currently Using:
PyTorch - For the sentiment analysis (Hugging Face transformers)
TensorFlow + Keras - Installed and available for custom models
Why PyTorch for this project? Pre-trained Hugging Face models work best with PyTorch.
