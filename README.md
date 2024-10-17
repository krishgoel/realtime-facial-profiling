# Facial Profiling Pipeline

This project implements a real-time facial recognition system integrated with servo-controlled camera tracking (Feetech STS3032).


![Facial Recognition Pipeline](./images/profiling-pipeline.png)


## Features

- Real-time face detection and tracking
- Servo-controlled camera movement
- Feature vector extraction for facial recognition
- Pinecone integration for vector similarity search
- MongoDB integration for user profiles
- Asynchronous processing for improved performance

## Quick Start

1. Clone the repository and install dependencies:
   ```
   git clone https://github.com/krishgoel/realtime-facial-profiling.git
   cd facial-profiling-pipeline
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   MONGO_URI=your_mongodb_uri
   ```

3. Run the application:
   ```
   python -m app
   ```

## Future Improvements

### Planned Enhancements
- Implement asynchronous streaming and processing for improved performance
- Containerize the entire pipeline using Docker for easier deployment and scalability

### Potential Optimizations
- Implement multi-vector storage for each individual to improve recognition accuracy
- Develop an adaptive frame capture system to account for appearance variations over time
- Implement periodic facial analysis updates (age, gender, race) with weighted averaging
- Enhance error handling for cases where no face is detected
- Improve matching accuracy by incorporating demographic analysis alongside facial vectors