import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def filter_and_load_data(file_path):
    """
    Reads data, saves a copy for inspection, provides diagnostics, and then filters the data.
    """
    try:
        # Build column names to keep extra columns that appear after the header.
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_fields = lines[0].split()
        max_len = max(len(line.split()) for line in lines)
        column_names = header_fields + [f'extra_{i}' for i in range(len(header_fields), max_len)]

        # Skip the first data row (index 1) to avoid misaligned entries below the header.
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=0,
            names=column_names,
            engine='python',
            skiprows=[1]
        )

        unfiltered_csv_path = 'read_data_for_inspection.csv'
        df.to_csv(unfiltered_csv_path, index=False)
        print(f"\nSaved the loaded table to '{unfiltered_csv_path}' for review.")
        print("--- Head of Loaded Table (before filtering) ---")
        print(df.head())
        print("-------------------------------------------------\n")

        filter_column_index = 12
        filter_column_name = df.columns[filter_column_index]
        df[filter_column_name] = pd.to_numeric(df[filter_column_name], errors='coerce')
        
        print("\n--- Diagnostics for Filtering Column ---")
        print(f"Reviewing statistics for column: '{filter_column_name}'")
        
        valid_data = df[filter_column_name].dropna()
        if valid_data.empty:
            print("The filtering column has no valid numbers.")
        else:
            print(valid_data.describe())
        print("------------------------------------\n")

        min_filter, max_filter = 2.0, 4.0
        print(f"Applying filter: Keeping rows where '{filter_column_name}' is between {min_filter} and {max_filter}...")

        filtered_df = df[
            (df[filter_column_name] >= min_filter) & (df[filter_column_name] <= max_filter)
        ].copy()

        # Deduplicate elements: prefer plain names (no suffix) except Sn -> Sn_d, Ru -> Ru_pv.
        def dedupe_by_element(df_in):
            if df_in.empty:
                return df_in
            preferred_special = {'Sn': 'Sn_d', 'Ru': 'Ru_pv'}
            deduped_rows = []
            base_names = df_in['element'].apply(lambda x: str(x).split('_')[0]).unique()
            for base in base_names:
                subset = df_in[df_in['element'].str.startswith(base)]
                preferred_name = preferred_special.get(base, base)
                if preferred_name in subset['element'].values:
                    chosen = subset[subset['element'] == preferred_name].iloc[0]
                else:
                    chosen = subset.iloc[0]
                deduped_rows.append(chosen)
            return pd.DataFrame(deduped_rows)

        filtered_df = dedupe_by_element(filtered_df)

        print(f"\nOriginal row count: {len(df)}")
        print(f"Row count after filtering: {len(filtered_df)}")
        
        return filtered_df

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
    except IndexError:
        print(f"Error: Not enough columns in file. Check for at least {filter_column_index + 1} columns.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def create_correlation_plot(df):
    """
    Creates and saves a scatter plot with two datasets and their trendlines.
    """
    if df is None or df.empty:
        print("Input DataFrame is empty, cannot create plot.")
        return

    try:
        x_col_index, y1_col_index, y2_col_index = 6, 8, 7
        x_col_name, y1_col_name, y2_col_name = df.columns[x_col_index], df.columns[y1_col_index], df.columns[y2_col_index]
        
        for col in [x_col_name, y1_col_name, y2_col_name]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_plot1 = df[[x_col_name, y1_col_name]].dropna()
        df_plot2 = df[[x_col_name, y2_col_name]].dropna()

        print(f"Plotting '{y1_col_name}' vs. '{x_col_name}' ({len(df_plot1)} points).")
        print(f"Plotting '{y2_col_name}' vs. '{x_col_name}' ({len(df_plot2)} points).")

        plt.figure(figsize=(12, 8))

        if not df_plot1.empty:
            x1, y1 = df_plot1[x_col_name], df_plot1[y1_col_name]
            plt.scatter(x1, y1, alpha=0.6, label=f'{y1_col_name} vs. {x_col_name}')
            m1, b1 = np.polyfit(x1, y1, 1)
            plt.plot(x1, m1*x1 + b1, color='navy', linestyle='--', label=f'Trend for {y1_col_name}')

        if not df_plot2.empty:
            x2, y2 = df_plot2[x_col_name], df_plot2[y2_col_name]
            plt.scatter(x2, y2, alpha=0.6, marker='^', label=f'{y2_col_name} vs. {x_col_name}')
            m2, b2 = np.polyfit(x2, y2, 1)
            plt.plot(x2, m2*x2 + b2, color='darkgreen', linestyle='--', label=f'Trend for {y2_col_name}')

        plt.xlabel(f"Adsorption Energy: {x_col_name} (eV)")
        plt.ylabel("Adsorption Energy (eV)")
        plt.title("Correlation of Adsorption Energies")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        output_filename = 'adsorption_energy_correlation.png'
        plt.savefig(output_filename)
        
        print(f"\nPlot successfully generated and saved as '{output_filename}'.")

    except IndexError:
        print("Error: DataFrame lacks required columns.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == '__main__':
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d2.dat'
    filtered_dataframe = filter_and_load_data(DATA_FILE)
    create_correlation_plot(filtered_dataframe)
