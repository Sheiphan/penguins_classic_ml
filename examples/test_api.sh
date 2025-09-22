#!/bin/bash
# Simple shell script to test the Penguin Classifier API
# Usage: ./examples/test_api.sh [host] [port]

set -e

HOST=${1:-localhost}
PORT=${2:-8000}
BASE_URL="http://${HOST}:${PORT}"

echo "Testing Penguin Classifier API at ${BASE_URL}"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Test 1: Health Check
echo
print_info "Testing health check endpoint..."
if curl -s -f "${BASE_URL}/health" > /dev/null; then
    HEALTH_RESPONSE=$(curl -s "${BASE_URL}/health")
    echo "Response: ${HEALTH_RESPONSE}"
    print_status 0 "Health check passed"
else
    print_status 1 "Health check failed - is the API server running?"
    echo "Try running: make api"
    exit 1
fi

# Test 2: Model Info
echo
print_info "Testing model info endpoint..."
if curl -s -f "${BASE_URL}/model/info" > /dev/null; then
    MODEL_INFO=$(curl -s "${BASE_URL}/model/info")
    echo "Response: ${MODEL_INFO}"
    print_status 0 "Model info retrieved successfully"
else
    print_status 1 "Model info failed"
fi

# Test 3: Single Prediction - Adelie
echo
print_info "Testing single prediction (Adelie penguin)..."
ADELIE_DATA='{
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "MALE",
    "year": 2007
}'

if ADELIE_RESULT=$(curl -s -f -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${ADELIE_DATA}"); then
    echo "Input: ${ADELIE_DATA}"
    echo "Response: ${ADELIE_RESULT}"
    print_status 0 "Adelie prediction successful"
else
    print_status 1 "Adelie prediction failed"
fi

# Test 4: Single Prediction - Chinstrap
echo
print_info "Testing single prediction (Chinstrap penguin)..."
CHINSTRAP_DATA='{
    "island": "Dream",
    "bill_length_mm": 46.5,
    "bill_depth_mm": 17.9,
    "flipper_length_mm": 192.0,
    "body_mass_g": 3500.0,
    "sex": "FEMALE",
    "year": 2008
}'

if CHINSTRAP_RESULT=$(curl -s -f -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${CHINSTRAP_DATA}"); then
    echo "Input: ${CHINSTRAP_DATA}"
    echo "Response: ${CHINSTRAP_RESULT}"
    print_status 0 "Chinstrap prediction successful"
else
    print_status 1 "Chinstrap prediction failed"
fi

# Test 5: Single Prediction - Gentoo
echo
print_info "Testing single prediction (Gentoo penguin)..."
GENTOO_DATA='{
    "island": "Biscoe",
    "bill_length_mm": 48.6,
    "bill_depth_mm": 16.0,
    "flipper_length_mm": 230.0,
    "body_mass_g": 5800.0,
    "sex": "MALE",
    "year": 2009
}'

if GENTOO_RESULT=$(curl -s -f -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${GENTOO_DATA}"); then
    echo "Input: ${GENTOO_DATA}"
    echo "Response: ${GENTOO_RESULT}"
    print_status 0 "Gentoo prediction successful"
else
    print_status 1 "Gentoo prediction failed"
fi

# Test 6: Batch Prediction
echo
print_info "Testing batch prediction..."
BATCH_DATA='{
    "instances": [
        {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
            "year": 2007
        },
        {
            "island": "Dream",
            "bill_length_mm": 46.5,
            "bill_depth_mm": 17.9,
            "flipper_length_mm": 192.0,
            "body_mass_g": 3500.0,
            "sex": "FEMALE",
            "year": 2008
        },
        {
            "island": "Biscoe",
            "bill_length_mm": 48.6,
            "bill_depth_mm": 16.0,
            "flipper_length_mm": 230.0,
            "body_mass_g": 5800.0,
            "sex": "MALE",
            "year": 2009
        }
    ]
}'

if BATCH_RESULT=$(curl -s -f -X POST "${BASE_URL}/predict/batch" \
    -H "Content-Type: application/json" \
    -d "${BATCH_DATA}"); then
    echo "Response: ${BATCH_RESULT}"
    print_status 0 "Batch prediction successful"
else
    print_status 1 "Batch prediction failed"
fi

# Test 7: Error Handling
echo
print_info "Testing error handling (invalid island)..."
INVALID_DATA='{
    "island": "InvalidIsland",
    "bill_length_mm": 45.0,
    "bill_depth_mm": 16.0,
    "flipper_length_mm": 200.0,
    "body_mass_g": 4000.0,
    "sex": "MALE",
    "year": 2008
}'

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${INVALID_DATA}")

if [ "${HTTP_STATUS}" = "422" ]; then
    ERROR_RESPONSE=$(curl -s -X POST "${BASE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "${INVALID_DATA}")
    echo "Error response: ${ERROR_RESPONSE}"
    print_status 0 "Error handling working correctly (HTTP 422)"
else
    print_status 1 "Error handling failed - expected HTTP 422, got ${HTTP_STATUS}"
fi

echo
echo "================================================"
print_info "API testing completed!"
echo
echo "For more comprehensive testing, run:"
echo "  python examples/test_api.py"
echo
echo "For interactive testing, try:"
echo "  curl -X POST \"${BASE_URL}/predict\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"island\": \"Biscoe\", \"bill_length_mm\": 48.6, \"bill_depth_mm\": 16.0, \"flipper_length_mm\": 230.0, \"body_mass_g\": 5800.0, \"sex\": \"MALE\", \"year\": 2009}'"