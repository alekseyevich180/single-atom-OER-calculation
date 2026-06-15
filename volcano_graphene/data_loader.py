import pandas as pd

def filter_and_load_data(file_path):
    """
    Reads data, saves the read data to a CSV for inspection, and then filters it.
    This helps in debugging why the filtering might be removing all data.
    """
    try:
        # Build column names to keep extra columns that appear after the header.
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_fields = lines[0].split()
        max_len = max(len(line.split()) for line in lines)
        column_names = header_fields + [f'extra_{i}' for i in range(len(header_fields), max_len)]

        # Use sep='\s+' to handle whitespace-delimited files and avoid FutureWarning.
        # The 'python' engine is more robust for complex delimiters.
        # Skip the first data row (index 1) to avoid misaligned entries below the header.
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=0,
            names=column_names,
            engine='python',
            skiprows=[1]
        )

        # --- Save the unfiltered data for user inspection ---
        unfiltered_csv_path = 'read_data_for_inspection.csv'
        df.to_csv(unfiltered_csv_path, index=False)
        print(f"\nSaved the initially loaded table to '{unfiltered_csv_path}' for your review.")
        print("--- Head of Loaded Table (before filtering) ---")
        print(df.head())
        print("-------------------------------------------------\n")

        # --- Proceed with diagnostics and filtering ---
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
                    # Fallback to the first occurrence for that base.
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
        print(f"Error: The file does not have enough columns. Check that it has at least {filter_column_index + 1} columns.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # This block allows for direct testing of this script.
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d2.dat'
    
    print(f"--- Running Data Loader Test on {DATA_FILE} ---")
    filtered_data = filter_and_load_data(DATA_FILE)
    
    if filtered_data is not None and not filtered_data.empty:
        print("\nTest Run: Filtered Data Head:")
        print(filtered_data.head())
    elif filtered_data is not None:
        print("\nTest Run: No data remained after filtering.")
    else:
        print("\nTest Run: Failed to load or process data.")
