import psycopg2
import json
import os
import concurrent.futures
import threading
DB_CONFIG = {
    "dbname": "stackoverflow_dba",
    "user": "postgres",
    "password": "Ksp@9907",
    "host": "localhost",
    "options": "-c jit=off" 
}
QUERY_DIRECTORY = "v1.0/stackoverflow/queries"
OUTPUT_FILE = "ground_truth_dataset.json"
TIMEOUT_MS = 30000
ANALYZE_TIMEOUT_MS = 20000
MAX_COST_THRESHOLD = 1000000
file_lock = threading.Lock()
def append_to_json_file(result):
    with file_lock:
        if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
            with open(OUTPUT_FILE, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(result)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
def estimate_query_cost(cursor, query_text):
    """Get estimated cost without executing the query."""
    try:
        explain_query = f"EXPLAIN (FORMAT JSON) {query_text}"
        cursor.execute(explain_query)
        plan_json = cursor.fetchone()[0][0]
        total_cost = plan_json["Plan"].get("Total Cost", 0.0)
        planning_time = plan_json.get("Planning Time", None)
        node_type = plan_json["Plan"].get("Node Type", "Unknown")
        return total_cost, planning_time, node_type, plan_json["Plan"]
    except Exception as e:
        print(f"Error estimating cost: {e}")
        return None, None, None, None
def execute_with_analyze(cursor, query_text, timeout_ms):
    """Execute query with ANALYZE and return actual metrics."""
    try:
        cursor.execute(f"SET statement_timeout = {timeout_ms};")
        explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query_text}"
        cursor.execute(explain_query)
        plan_json = cursor.fetchone()[0][0]
        execution_time_ms = plan_json.get("Execution Time", 0.0)
        planning_time_ms = plan_json.get("Planning Time", None)
        total_cost = plan_json["Plan"].get("Total Cost", 0.0)
        node_type = plan_json["Plan"].get("Node Type", "Unknown")
        return {
            "execution_time_ms": execution_time_ms,
            "planning_time_ms": planning_time_ms,
            "total_cost": total_cost,
            "node_type": node_type,
            "is_analyzed": True,
            "time_is_estimated": False
        }
    except psycopg2.errors.QueryCanceled:
        print(f"Query exceeded {timeout_ms}ms timeout during ANALYZE")
        return None
    except Exception as e:
        print(f"Error during ANALYZE execution: {e}")
        return None
def get_query_time(filename):
    local_conn = None
    local_cursor = None
    try:
        local_conn = psycopg2.connect(**DB_CONFIG)
        local_conn.autocommit = True
        local_cursor = local_conn.cursor()
        file_path = os.path.join(QUERY_DIRECTORY, filename)
        with open(file_path, 'r') as f:
            query_text = f.read().strip()
        print(f"Estimating cost for {filename}...")
        estimated_cost, planning_time, node_type, plan = estimate_query_cost(local_cursor, query_text)
        if estimated_cost is None:
            return None
        result = {
            "query_id": filename,
            "total_cost": estimated_cost,
            "is_analyzed": False,
            "planning_time_ms": planning_time,
            "node_type": node_type,
            "time_is_estimated": True
        }
        if estimated_cost > MAX_COST_THRESHOLD:
            print(f"  {filename} has high estimated cost ({estimated_cost:.2f}), skipping ANALYZE")
            result["execution_time_ms"] = estimated_cost / 100.0
        else:
            print(f"Running ANALYZE for {filename} (estimated cost: {estimated_cost:.2f})...")
            analyze_result = execute_with_analyze(local_cursor, query_text, ANALYZE_TIMEOUT_MS)
            if analyze_result:
                result.update(analyze_result)
                print(f" {filename}: {result['execution_time_ms']:.2f}ms (Cost: {result['total_cost']:.2f})")
            else:
                print(f"  {filename} timed out during ANALYZE, using estimate")
                result["execution_time_ms"] = estimated_cost / 100.0
        append_to_json_file(result)
        return result
    except Exception as e:
        print(f" Error processing {filename}: {e}")
        return None
    finally:
        if local_cursor:
            local_cursor.close()
        if local_conn:
            local_conn.close()
if __name__ == "__main__":
    with open(OUTPUT_FILE, 'w') as f:
        json.dump([], f)
    if not os.path.exists(QUERY_DIRECTORY):
        print(f"Directory '{QUERY_DIRECTORY}' not found.")
        list_of_filenames = []
    else:
        list_of_filenames = sorted([f for f in os.listdir(QUERY_DIRECTORY) if f.endswith(".sql")])
    print(f"Starting parallel execution for {len(list_of_filenames)} queries...")
    print(f"Max Cost Threshold: {MAX_COST_THRESHOLD}")
    print(f"ANALYZE Timeout: {ANALYZE_TIMEOUT_MS}ms\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_filename = {executor.submit(get_query_time, fn): fn for fn in list_of_filenames}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_filename):
            completed += 1
            filename = future_to_filename[future]
            try:
                result = future.result()
                if result:
                    print(f"[{completed}/{len(list_of_filenames)}] Completed {filename}")
            except Exception as e:
                print(f"[{completed}/{len(list_of_filenames)}] Failed {filename}: {e}")
    print(f"\nSuccessfully processed queries. Results saved to {OUTPUT_FILE}")