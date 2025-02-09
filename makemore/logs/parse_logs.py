import re
from pprint import pprint

def parse_logs(filename):
    """
    Parse hyperparameter search logs and return a dictionary mapping
    each configuration header to a list of run details.
    """
    configs = {}
    current_config = None

    # Read the entire log file into lines
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect a configuration header line.
        if line.startswith("HYPERPARAMETERS SEARCH, START"):
            # Extract configuration details from the header
            # e.g., "HYPERPARAMETERS SEARCH, START block_size_4__embedding_size_20__..."
            current_config = line.split("HYPERPARAMETERS SEARCH, START", 1)[1].strip()
            # Initialize a list for this configuration if it doesn't exist.
            if current_config not in configs:
                configs[current_config] = []
            i += 1
            continue
        
        # Detect a "starting run:" line.
        if line.startswith("starting run:"):
            # Extract the run's full configuration (including run_idx)
            run_line = line.split("starting run:", 1)[1].strip()
            
            # Use regex to extract the run index after "__run_idx_"
            run_idx_match = re.search(r"__run_idx_(\d+)", run_line)
            run_idx = int(run_idx_match.group(1)) if run_idx_match else None
            
            # For this example, assume the next few lines contain:
            #   line: start value (e.g., "0")
            #   line: end value (e.g., "10000")
            #   line: "train() COMPLETED"
            #   line: first metric (e.g., training loss)
            #   line: second metric (e.g., evaluation loss)
            try:
                start_value = float(lines[i+1])
                end_value   = float(lines[i+2])
                # Ensure we have the expected marker:
                if lines[i+3] != "train() COMPLETED":
                    print(f"Unexpected marker at line {i+3}: {lines[i+3]}")
                metric1 = float(lines[i+4])
                metric2 = float(lines[i+5])
            except (IndexError, ValueError) as e:
                print("Error parsing run details at line", i, e)
                break

            run_info = {
                "run_idx": run_idx,
                "start": start_value,
                "end": end_value,
                "metric1": metric1,
                "metric2": metric2,
                "full_run_config": run_line,
            }
            
            # Append the run info under the current configuration header.
            if current_config is not None:
                configs[current_config].append(run_info)
            else:
                print("Warning: Found a run without a preceding configuration header.")
            
            # Skip ahead past the lines we just parsed.
            i += 6
            continue
        
        # If the line doesn't match any known pattern, skip it.
        i += 1
    
    return configs


if __name__ == "__main__":
    # Change the filename to your log file's path.
    filename = "logs.txt"
    parsed_configs = parse_logs(filename)
    
    # Pretty-print the parsed log results.
    pprint(parsed_configs)

