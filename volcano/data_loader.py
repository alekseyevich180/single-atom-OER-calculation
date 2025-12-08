import pandas as pd

def filter_and_load_data(file_path):
    """
    Reads data from the specified file and filters it.
    
    - Filters for rows where the value in column 13 (index 12) is between 2.0 and 4.0.
    - Returns a pandas DataFrame with the filtered data.
    """
    try:
        # Use sep='\s+' instead of delim_whitespace=True to avoid FutureWarning
        # The regex engine is specified to handle potential parsing issues with this format.
        df = pd.read_csv(file_path, sep=r'\s+', header=0, engine='python')

        filter_column_name = df.columns[12]
        df[filter_column_name] = pd.to_numeric(df[filter_column_name], errors='coerce')
        
        # --- Diagnostic Print ---
        # Show statistics of the column used for filtering BEFORE the filter is applied
        print("\n--- Diagnostics for Filtering Column ---")
        print(f"Statistics for column '{filter_column_name}':")
        
        # Drop NaN values for accurate description
        valid_data = df[filter_column_name].dropna()
        if valid_data.empty:
            print("The filtering column contains no valid numeric data.")
        else:
            print(valid_data.describe())
        print("------------------------------------\n")

        # The user-defined filter range
        min_filter, max_filter = 2.0, 4.0
        print(f"Filtering rows where '{filter_column_name}' is between {min_filter} and {max_filter}...")

        filtered_df = df[
            (df[filter_column_name] >= min_filter) & (df[filter_column_name] <= max_filter)
        ].copy()
        
        print(f"Successfully loaded data from {file_path}.")
        print(f"Original rows: {len(df)}, Rows after filtering: {len(filtered_df)}")
        
        return filtered_df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except IndexError:
        print("Error: The file does not have enough columns for filtering.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # This is for testing the data loader directly
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d2.dat'
    filtered_data = filter_and_load_data(DATA_FILE)
    
    if filtered_data is not None and not filtered_data.empty:
        print("\nFiltered Data Head:")
        print(filtered_data.head())
    elif filtered_data is not None:
        print("\nNo data remains after filtering.")