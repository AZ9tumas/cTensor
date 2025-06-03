import csv
import argparse
import sys
import os

def parse_reports(report_paths):
    """Parses cTensor test reports and identifies failures."""
    all_failures = []
    passed_all_reports = True

    print(f"Checking reports: {report_paths}")

    for report_path in report_paths:
        if not os.path.exists(report_path):
            print(f"Error: Report file not found: {report_path}", file=sys.stderr)
            passed_all_reports = False
            all_failures.append({
                "file": os.path.basename(report_path),
                "operator": "N/A",
                "test_point": "FILE_NOT_FOUND",
                "details": f"Report file {report_path} was not found."
            })
            continue
        
        try:
            with open(report_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                # Check for essential headers
                expected_headers = ['Operator', 'TestPoint', 'ResultDetail']
                if not reader.fieldnames or not all(header in reader.fieldnames for header in expected_headers):
                    print(f"Error: Report file {report_path} has missing or incorrect headers.", file=sys.stderr)
                    print(f"Expected headers: {expected_headers}, Got: {reader.fieldnames}", file=sys.stderr)
                    passed_all_reports = False
                    all_failures.append({
                        "file": os.path.basename(report_path),
                        "operator": "N/A",
                        "test_point": "INVALID_HEADERS",
                        "details": f"Report file {report_path} has invalid or missing CSV headers."
                    })
                    continue

                report_has_failures = False
                print(f"Processing report: {report_path}")
                for row_num, row in enumerate(reader, 1):
                    operator = row.get('Operator', 'N/A')
                    test_point = row.get('TestPoint', 'N/A')
                    result_detail = row.get('ResultDetail', 'ERROR_MISSING_RESULT_DETAIL')

                    if result_detail != '/':
                        passed_all_reports = False
                        report_has_failures = True
                        all_failures.append({
                            "file": os.path.basename(report_path),
                            "operator": operator,
                            "test_point": test_point,
                            "details": result_detail
                        })
                if report_has_failures:
                    print(f"Failures found in {os.path.basename(report_path)}.", file=sys.stderr)
                else:
                    print(f"No failures found in {os.path.basename(report_path)}.")

        except Exception as e:
            print(f"Error reading or parsing {report_path}: {e}", file=sys.stderr)
            passed_all_reports = False
            all_failures.append({
                "file": os.path.basename(report_path),
                "operator": "N/A",
                "test_point": "PARSING_ERROR",
                "details": f"Could not parse {report_path}. Error: {e}"
            })

    return passed_all_reports, all_failures

def main():
    parser = argparse.ArgumentParser(description="Parse cTensor test reports and check for failures.")
    parser.add_argument("report_files", nargs='+', help="Paths to the cten_test_report.csv files.")
    args = parser.parse_args()

    passed_all, failures = parse_reports(args.report_files)

    if not passed_all:
        print("\n--- Test Failures Summary ---", file=sys.stderr)
        for failure in failures:
            print(f"  File: {failure['file']}, Operator: {failure['operator']}, Test: {failure['test_point']}, Details: {failure['details']}", file=sys.stderr)
        print("\nOne or more tests failed or reports were invalid.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll tests passed across all reports.")
        sys.exit(0)

if __name__ == "__main__":
    main()
