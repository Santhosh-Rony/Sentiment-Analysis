"""
Main entry point for the sentiment analysis project.
"""

import argparse
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api_server(args):
    """Run the FastAPI server."""
    logger.info("Starting FastAPI server...")
    try:
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        raise

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis API with Pre-trained DistilBERT'
    )
    
    # Add server arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number (default: 8000)')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Log level (default: info)')
    
    args = parser.parse_args()
    run_api_server(args)

if __name__ == "__main__":
    main()