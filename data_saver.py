# data_saver.py
import pandas as pd
import os

class DataSaver:
    def __init__(self, csv_path, failed_path):
        self.csv_path = csv_path
        self.failed_path = failed_path
        self.warnings_path = csv_path.replace(".csv", "_warnings.txt")
        self.df = pd.DataFrame()
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            if 'Filename' not in self.df.columns:
                self.df = pd.DataFrame(columns=['Filename'])
        else:
            self.df = pd.DataFrame(columns=['Filename'])

    def add_data(self, data):
        new_row = pd.DataFrame([data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)

    def log_failure(self, refcode, error_message):
        with open(self.failed_path, 'a') as file:
            file.write(f"{refcode}: {error_message}\n")

    def log_warning(self, refcode, warning_message):
        with open(self.warnings_path, 'a') as file:
            file.write(f"{refcode}: {warning_message}\n")
