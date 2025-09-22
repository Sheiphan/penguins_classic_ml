#!/usr/bin/env python3
"""
Test script for the Enterprise ML Classifier API.

This script demonstrates how to interact with the API endpoints
and provides examples of different types of requests.

Usage:
    python examples/test_api.py [--host localhost] [--port 8000]

Requirements:
    - API server running (make api)
    - requests library (pip install requests)
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List

import requests
from requests.exceptions import ConnectionError, RequestException


class PenguinClassifierClient:
    """Client for interacting with the Penguin Classifier API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health status.
        
        Returns:
            Health check response
            
        Raises:
            RequestException: If request fails
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information response
            
        Raises:
            RequestException: If request fails
        """
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, penguin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction.
        
        Args:
            penguin_data: Dictionary containing penguin features
            
        Returns:
            Prediction response
            
        Raises:
            RequestException: If request fails
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=penguin_data
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, penguin_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions.
        
        Args:
            penguin_instances: List of penguin feature dictionaries
            
        Returns:
            Batch prediction response
            
        Raises:
            RequestException: If request fails
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"instances": penguin_instances}
        )
        response.raise_for_status()
        return response.json()


def load_sample_data() -> Dict[str, Any]:
    """Load sample penguin data for testing.
    
    Returns:
        Dictionary containing sample requests
    """
    return {
        "adelie_male": {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
            "year": 2007
        },
        "chinstrap_female": {
            "island": "Dream",
            "bill_length_mm": 46.5,
            "bill_depth_mm": 17.9,
            "flipper_length_mm": 192.0,
            "body_mass_g": 3500.0,
            "sex": "FEMALE",
            "year": 2008
        },
        "gentoo_male": {
            "island": "Biscoe",
            "bill_length_mm": 48.6,
            "bill_depth_mm": 16.0,
            "flipper_length_mm": 230.0,
            "body_mass_g": 5800.0,
            "sex": "MALE",
            "year": 2009
        }
    }


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_json(data: Dict[str, Any], title: str = "") -> None:
    """Print JSON data in a formatted way."""
    if title:
        print(f"\n{title}:")
    print(json.dumps(data, indent=2))


def test_health_check(client: PenguinClassifierClient) -> bool:
    """Test the health check endpoint.
    
    Args:
        client: API client instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        print_section("Health Check Test")
        health = client.health_check()
        print_json(health, "Health Status")
        
        if health.get("status") == "healthy":
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed - service not healthy")
            return False
            
    except RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_model_info(client: PenguinClassifierClient) -> bool:
    """Test the model info endpoint.
    
    Args:
        client: API client instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        print_section("Model Info Test")
        model_info = client.get_model_info()
        print_json(model_info, "Model Information")
        
        required_fields = ["model_name", "features", "target_classes"]
        missing_fields = [field for field in required_fields if field not in model_info]
        
        if not missing_fields:
            print("✅ Model info test passed")
            return True
        else:
            print(f"❌ Model info test failed - missing fields: {missing_fields}")
            return False
            
    except RequestException as e:
        print(f"❌ Model info test failed: {e}")
        return False


def test_single_predictions(client: PenguinClassifierClient) -> bool:
    """Test single prediction endpoints.
    
    Args:
        client: API client instance
        
    Returns:
        True if all tests pass, False otherwise
    """
    print_section("Single Prediction Tests")
    sample_data = load_sample_data()
    all_passed = True
    
    for name, penguin_data in sample_data.items():
        try:
            print(f"\nTesting {name}:")
            print_json(penguin_data, "Input")
            
            result = client.predict_single(penguin_data)
            print_json(result, "Prediction")
            
            # Validate response structure
            required_fields = ["prediction"]
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                all_passed = False
            elif result["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]:
                print(f"✅ Prediction successful: {result['prediction']}")
            else:
                print(f"❌ Invalid prediction: {result['prediction']}")
                all_passed = False
                
        except RequestException as e:
            print(f"❌ Prediction failed for {name}: {e}")
            all_passed = False
    
    return all_passed


def test_batch_predictions(client: PenguinClassifierClient) -> bool:
    """Test batch prediction endpoint.
    
    Args:
        client: API client instance
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        print_section("Batch Prediction Test")
        sample_data = load_sample_data()
        instances = list(sample_data.values())
        
        print_json({"instances": instances}, "Batch Input")
        
        result = client.predict_batch(instances)
        print_json(result, "Batch Predictions")
        
        # Validate response structure
        if "predictions" not in result:
            print("❌ Missing 'predictions' field in response")
            return False
        
        if len(result["predictions"]) != len(instances):
            print(f"❌ Expected {len(instances)} predictions, got {len(result['predictions'])}")
            return False
        
        # Validate each prediction
        for i, prediction in enumerate(result["predictions"]):
            if "prediction" not in prediction:
                print(f"❌ Missing 'prediction' field in result {i}")
                return False
            
            if prediction["prediction"] not in ["Adelie", "Chinstrap", "Gentoo"]:
                print(f"❌ Invalid prediction in result {i}: {prediction['prediction']}")
                return False
        
        print("✅ Batch prediction test passed")
        return True
        
    except RequestException as e:
        print(f"❌ Batch prediction test failed: {e}")
        return False


def test_error_handling(client: PenguinClassifierClient) -> bool:
    """Test API error handling with invalid inputs.
    
    Args:
        client: API client instance
        
    Returns:
        True if error handling works correctly, False otherwise
    """
    print_section("Error Handling Tests")
    
    # Test cases with expected errors
    error_cases = [
        {
            "name": "Invalid island",
            "data": {
                "island": "InvalidIsland",
                "bill_length_mm": 45.0,
                "bill_depth_mm": 16.0,
                "flipper_length_mm": 200.0,
                "body_mass_g": 4000.0,
                "sex": "MALE",
                "year": 2008
            },
            "expected_status": 422
        },
        {
            "name": "Negative bill length",
            "data": {
                "island": "Biscoe",
                "bill_length_mm": -5.0,
                "bill_depth_mm": 16.0,
                "flipper_length_mm": 200.0,
                "body_mass_g": 4000.0,
                "sex": "MALE",
                "year": 2008
            },
            "expected_status": 422
        },
        {
            "name": "Missing required field",
            "data": {
                "island": "Biscoe",
                "bill_depth_mm": 16.0,
                "flipper_length_mm": 200.0,
                "body_mass_g": 4000.0,
                "sex": "MALE",
                "year": 2008
                # Missing bill_length_mm
            },
            "expected_status": 422
        }
    ]
    
    all_passed = True
    
    for case in error_cases:
        try:
            print(f"\nTesting {case['name']}:")
            print_json(case['data'], "Invalid Input")
            
            response = client.session.post(
                f"{client.base_url}/predict",
                json=case['data']
            )
            
            if response.status_code == case['expected_status']:
                error_detail = response.json()
                print_json(error_detail, "Error Response")
                print(f"✅ Error handling correct (status {response.status_code})")
            else:
                print(f"❌ Expected status {case['expected_status']}, got {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error handling test failed for {case['name']}: {e}")
            all_passed = False
    
    return all_passed


def test_performance(client: PenguinClassifierClient, num_requests: int = 10) -> bool:
    """Test API performance with multiple requests.
    
    Args:
        client: API client instance
        num_requests: Number of requests to make
        
    Returns:
        True if performance is acceptable, False otherwise
    """
    print_section(f"Performance Test ({num_requests} requests)")
    
    sample_data = load_sample_data()
    test_penguin = list(sample_data.values())[0]
    
    response_times = []
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            result = client.predict_single(test_penguin)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            successful_requests += 1
            
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{num_requests} requests")
                
        except RequestException as e:
            print(f"Request {i + 1} failed: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\nPerformance Results:")
        print(f"  Successful requests: {successful_requests}/{num_requests}")
        print(f"  Average response time: {avg_time:.3f}s")
        print(f"  Min response time: {min_time:.3f}s")
        print(f"  Max response time: {max_time:.3f}s")
        
        # Consider performance acceptable if average < 1s and success rate > 95%
        success_rate = successful_requests / num_requests
        performance_ok = avg_time < 1.0 and success_rate > 0.95
        
        if performance_ok:
            print("✅ Performance test passed")
        else:
            print("❌ Performance test failed (slow response or low success rate)")
        
        return performance_ok
    else:
        print("❌ No successful requests in performance test")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test the Penguin Classifier API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests for performance test")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    client = PenguinClassifierClient(base_url)
    
    print(f"Testing Penguin Classifier API at {base_url}")
    
    # Check if API is accessible
    try:
        client.session.get(f"{base_url}/health", timeout=5)
    except ConnectionError:
        print(f"❌ Cannot connect to API at {base_url}")
        print("Make sure the API server is running (try: make api)")
        sys.exit(1)
    
    # Run test suite
    tests = [
        ("Health Check", lambda: test_health_check(client)),
        ("Model Info", lambda: test_model_info(client)),
        ("Single Predictions", lambda: test_single_predictions(client)),
        ("Batch Predictions", lambda: test_batch_predictions(client)),
        ("Error Handling", lambda: test_error_handling(client)),
    ]
    
    if args.performance:
        tests.append(("Performance", lambda: test_performance(client, args.num_requests)))
    
    # Execute tests
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("Test Summary")
    passed_tests = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The API is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()