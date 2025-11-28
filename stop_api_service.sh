#!/bin/bash

echo "======================================"
echo "Stopping Face Recognition Services"
echo "======================================"


if [ -f dev/logs/detector.pid ]; then
    DETECTOR_PID=$(cat dev/logs/detector.pid)
    echo "Stopping Detector (PID: $DETECTOR_PID)..."
    kill $DETECTOR_PID 2>/dev/null && echo "  ✓ Detector stopped" || echo "  ✗ Detector not running"
    rm -f dev/logs/detector.pid
fi

if [ -f dev/logs/alignment.pid ]; then
    ALIGNMENT_PID=$(cat dev/logs/alignment.pid)
    echo "Stopping Alignment (PID: $ALIGNMENT_PID)..."
    kill $ALIGNMENT_PID 2>/dev/null && echo "  ✓ Alignment stopped" || echo "  ✗ Alignment not running"
    rm -f dev/logs/alignment.pid
fi

if [ -f dev/logs/recognition_1.pid ]; then
    RECOGNITION_PID_1=$(cat dev/logs/recognition_1.pid)
    echo "Stopping Recognition 1 (PID: $RECOGNITION_PID_1)..."
    kill $RECOGNITION_PID_1 2>/dev/null && echo "  ✓ Recognition 1 stopped" || echo "  ✗ Recognition 1 not running"
    rm -f dev/logs/recognition_1.pid
fi

if [ -f dev/logs/recognition_2.pid ]; then
    RECOGNITION_PID_2=$(cat dev/logs/recognition_2.pid)
    echo "Stopping Recognition 2 (PID: $RECOGNITION_PID_2)..."
    kill $RECOGNITION_PID_2 2>/dev/null && echo "  ✓ Recognition 2 stopped" || echo "  ✗ Recognition 2 not running"
    rm -f dev/logs/recognition_2.pid
fi

if [ -f dev/logs/recognition.pid ]; then
    RECOGNITION_PID=$(cat dev/logs/recognition.pid)
    echo "Stopping Legacy Recognition (PID: $RECOGNITION_PID)..."
    kill $RECOGNITION_PID 2>/dev/null && echo "  ✓ Legacy Recognition stopped" || echo "  ✗ Legacy Recognition not running"
    rm -f dev/logs/recognition.pid
fi

if [ -f dev/logs/main.pid ]; then
    MAIN_PID=$(cat dev/logs/main.pid)
    echo "Stopping Main (PID: $MAIN_PID)..."
    kill $MAIN_PID 2>/dev/null && echo "  ✓ Main stopped" || echo "  ✗ Main not running"
    rm -f dev/logs/main.pid
fi

echo ""
echo "======================================"
echo "All services stopped"
echo "======================================"
