#!/bin/bash


TESTS_DIR="unittests"

# Optional: Set the Python interpreter to use.
PYTHON=${PYTHON:-python3}

# Optional: Set to true to run tests in parallel. Requires GNU parallel.
PARALLEL=${PARALLEL:-false}

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Run a single test file.
run_test() {
    local test_file=$1
    local framework=$(basename $(dirname $test_file))
    
    echo -e "=== Running ${GREEN}$framework${NC} tests: ${test_file} ==="
    if $PYTHON $test_file; then
        echo -e "${GREEN}[PASSED]${NC} $test_file\n"
        return 0
    else
        echo -e "${RED}[FAILED]${NC} $test_file\n"
        return 1
    fi
}

# Main function to run all tests.
main() {
    local total=0
    local passed=0
    
    declare -a test_files=($(find $TESTS_DIR -type f -name "test_*.py" -o -name "*_test.py"))
    
    if [ ${#test_files[@]} -eq 0 ]; then
        echo -e "${RED}Error: No test files found in $TESTS_DIR subdirectories${NC}"
        exit 1
    fi

    if [ "$PARALLEL" = true ]; then
        if ! command -v parallel &> /dev/null; then
            echo -e "${RED}Error: GNU parallel is required for parallel execution${NC}"
            exit 1
        fi
        printf "%s\n" "${test_files[@]}" | parallel -j $(nproc) run_test
        return
    fi

    for test_file in "${test_files[@]}"; do
        ((total++))
        if run_test "$test_file"; then
            ((passed++))
        fi
    done

    echo -e "\n=== Test Summary ==="
    echo -e "Total: $total | ${GREEN}Passed: $passed${NC} | ${RED}Failed: $((total - passed))${NC}"
    
    [ $passed -eq $total ] && exit 0 || exit 1
}

main