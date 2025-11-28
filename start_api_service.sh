#!/bin/bash

echo "======================================"
echo "Starting Face Recognition Services"
echo "======================================"

DETECTOR_PORT=8001
ALIGNMENT_PORT=8002
RECOGNITION_PORT_1=8003
RECOGNITION_PORT_2=8005
MAIN_PORT=8004

mkdir -p dev/logs

echo ""
echo "1. Starting Detector API on port $DETECTOR_PORT..."
uvicorn product.api.detector_api:app --host 0.0.0.0 --port $DETECTOR_PORT > dev/logs/detector.log 2>&1 &
DETECTOR_PID=$!
echo "   ✓ Detector PID: $DETECTOR_PID"
echo $DETECTOR_PID > dev/logs/detector.pid

echo ""
echo "2. Starting Alignment API on port $ALIGNMENT_PORT..."
uvicorn product.api.alignment_api:app --host 0.0.0.0 --port $ALIGNMENT_PORT > dev/logs/alignment.log 2>&1 &
ALIGNMENT_PID=$!
echo "   ✓ Alignment PID: $ALIGNMENT_PID"
echo $ALIGNMENT_PID > dev/logs/alignment.pid

echo ""
echo "3. Starting Recognition API 1 on port $RECOGNITION_PORT_1..."
uvicorn product.api.recognition_api:app --host 0.0.0.0 --port $RECOGNITION_PORT_1 > dev/logs/recognition_1.log 2>&1 &
RECOGNITION_PID_1=$!
echo "   ✓ Recognition PID: $RECOGNITION_PID_1"
echo $RECOGNITION_PID_1 > dev/logs/recognition_1.pid

echo ""
echo "4. Starting Recognition API 2 on port $RECOGNITION_PORT_2..."
uvicorn product.api.recognition_api:app --host 0.0.0.0 --port $RECOGNITION_PORT_2 > dev/logs/recognition_2.log 2>&1 &
RECOGNITION_PID_2=$!
echo "   ✓ Recognition PID: $RECOGNITION_PID_2"
echo $RECOGNITION_PID_2 > dev/logs/recognition_2.pid

echo ""
echo "5. Starting Main API on port $MAIN_PORT..."
uvicorn product.api.main_api:app --host 0.0.0.0 --port $MAIN_PORT > dev/logs/main.log 2>&1 &
MAIN_PID=$!
echo "   ✓ MAIN PID: $MAIN_PID"
echo $MAIN_PID > dev/logs/main.pid



echo ""
echo "======================================"
echo "All services started successfully!"
echo "======================================"
echo "Detector:    http://localhost:$DETECTOR_PORT"
echo "Alignment:   http://localhost:$ALIGNMENT_PORT"
echo "Recognition 1: http://localhost:$RECOGNITION_PORT_1"
echo "Recognition 2: http://localhost:$RECOGNITION_PORT_2"
echo "Main:        http://localhost:$MAIN_PORT"
echo ""
echo "PIDs saved to logs/*.pid"
echo "Logs saved to logs/*.log"
echo ""
echo "To stop all services, run: ./stop_services.sh"
echo "======================================"
