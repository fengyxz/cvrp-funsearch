import csv
import uuid
import os
import logging

# Configure the log format and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def save_results_to_csv(results, folder="results"):
    """
    Save the experimental evaluation results to a CSV file.

    :param results: List[Dict], a list of results for each experiment, where each result is a dictionary.
    :param folder: str, optional, the folder to save the file, the default is the "results" folder.
    """
    # Check if the results are empty
    if not results:
        logging.warning("No results to save. The results list is empty.")
        return

    # Extract the field names from the result dictionary
    fieldnames = results[0].keys()

    # Create the folder to save the file (if it doesn't exist)
    if not os.path.exists(folder):
        os.makedirs(folder)
        logging.info(f"Created directory: {folder}")

    # Generate the file name using uuid
    filename = f"{folder}/results_{uuid.uuid4()}.csv"

    # Start writing to the file
    logging.info("Starting to write results to CSV file...")
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # Write the header
            writer.writeheader()
            # Write the data rows
            writer.writerows(results)

        # Finished writing
        logging.info(f"Results successfully saved to {filename}")
    except Exception as e:
        logging.error(f"An error occurred while writing to the file: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Evaluation results of the experiment, each result is a dictionary
    results = [
        {"is_success": True, "distance": 5.5},
        {"is_success": False, "distance": 12.3},
        {"is_success": True, "distance": 7.8}
    ]

    # Save the results to a CSV file
    save_results_to_csv(results)