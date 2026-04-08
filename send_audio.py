#!/usr/bin/env python3
"""
Script to send audio file to transcription endpoint and track processing time
Supports sending multiple concurrent requests
"""

import requests
import time
import sys
import threading
from pathlib import Path
from typing import List, Dict

results = []
results_lock = threading.Lock()

def send_single_request(request_id: int, file_path: str, endpoint_url: str):
    """
    Send a single audio file request in a thread
    
    Args:
        request_id: ID of the request
        file_path: Path to the audio file
        endpoint_url: URL of the endpoint
    """
    result = {
        'request_id': request_id,
        'file_path': file_path,
        'status': 'pending',
        'processing_time': 0,
        'status_code': None,
        'response': None,
        'error': None
    }
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            
            print(f"[Request #{request_id}] ⏱️  Starting upload and processing...")
            start_time = time.perf_counter()
            
            # Send POST request
            response = requests.post(endpoint_url, files=files, timeout=300)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
        result['status'] = 'completed'
        result['processing_time'] = processing_time
        result['status_code'] = response.status_code
        
        if response.status_code == 200:
            try:
                result['response'] = response.json()
            except:
                result['response'] = response.text[:500]
        else:
            result['error'] = response.text[:500]
            
    except requests.exceptions.Timeout:
        result['status'] = 'failed'
        result['error'] = 'Request timed out after 300 seconds'
    except requests.exceptions.ConnectionError:
        result['status'] = 'failed'
        result['error'] = f'Failed to connect to {endpoint_url}'
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
    
    # Thread-safe append to results
    with results_lock:
        results.append(result)

def send_audio_files(file_path: str, endpoint_url: str, num_requests: int = 2):
    """
    Send audio file multiple times concurrently and track processing time
    
    Args:
        file_path: Path to the audio file
        endpoint_url: URL of the endpoint to send the file to
        num_requests: Number of concurrent requests to send
    """
    # Validate file exists
    audio_file = Path(file_path)
    if not audio_file.exists():
        print(f"❌ Error: Audio file not found at {file_path}")
        sys.exit(1)
    
    print("🎵 Audio File Upload Script (Concurrent Requests)")
    print("=" * 70)
    print(f"📁 File: {file_path}")
    print(f"📏 File size: {audio_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"🔗 Endpoint: {endpoint_url}")
    print(f"🔄 Number of concurrent requests: {num_requests}")
    print("-" * 70)
    
    # Create and start threads
    threads = []
    overall_start = time.perf_counter()
    
    for i in range(num_requests):
        thread = threading.Thread(
            target=send_single_request,
            args=(i + 1, file_path, endpoint_url)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    overall_end = time.perf_counter()
    overall_time = overall_end - overall_start
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    for result in results:
        req_id = result['request_id']
        print(f"\n[Request #{req_id}]")
        print(f"  Status: {result['status']}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        if result['status_code']:
            print(f"  Status code: {result['status_code']}")
        
        if result['error']:
            print(f"  ❌ Error: {result['error']}")
        elif result['response']:
            print(f"  ✅ Response: {result['response']}")
    
    print("\n" + "-" * 70)
    print(f"⏱️  Overall time for {num_requests} concurrent requests: {overall_time:.2f}s")
    print(f"📈 Average time per request: {overall_time / num_requests:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    FILE_PATH = "./Is King Charles 5p Worthless-English Listening B2-C1 Ep 836-155c0f.mp3"
    ENDPOINT_URL = "http://13.201.65.162:8001/process/transcription"
    NUM_REQUESTS = 2
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        ENDPOINT_URL = sys.argv[2]
    if len(sys.argv) > 3:
        NUM_REQUESTS = int(sys.argv[3])
    
    send_audio_files(FILE_PATH, ENDPOINT_URL, NUM_REQUESTS)

