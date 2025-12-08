import pandas as pd
import matplotlib.pyplot as plt

def filter_and_plot_data(file_path):
    """
    Reads data from the specified file, filters it, and creates a volcano plot.
    
    - Filters for rows where the value in column 13 (index 12) is between 2.0 and 4.0.
    - Plots column 22 ('potential') vs. column 8 ('deltaE-O').
    """
    try:
        # Use read_csv with whitespace delimiter and treat the first row as a header.
        df = pd.read_csv(file_path, delim_whitespace=True, header=0)

        # --- 1. Filtering Step ---
        # Get the name of the 13th column (index 12) for filtering
        filter_column_name = df.columns[12]
        
        # Convert the filter column to a numeric type, coercing errors to NaN
        df[filter_column_name] = pd.to_numeric(df[filter_column_name], errors='coerce')
        
        # Apply the filter condition
        filtered_df = df[
            (df[filter_column_name] >= 2.0) & (df[filter_column_name] <= 4.0)
        ].copy()
        
        print(f"Successfully loaded data from {file_path}.")
        print(f"Original rows: {len(df)}, Rows after filtering: {len(filtered_df)}")

        if filtered_df.empty:
            print("No data remains after filtering. Cannot create a plot.")
            return

        # --- 2. Plotting Step ---
        # Define column indices for plotting
        x_col_index = 7  # Corresponds to 8th column 'deltaE-O'
        y_col_index = 21 # Corresponds to 22nd column 'potential'

        # Get column names
        x_col_name = filtered_df.columns[x_col_index]
        y_col_name = filtered_df.columns[y_col_index]
        
        # Convert plotting columns to numeric, coercing errors
        filtered_df[x_col_name] = pd.to_numeric(filtered_df[x_col_name], errors='coerce')
        filtered_df[y_col_name] = pd.to_numeric(filtered_df[y_col_name], errors='coerce')

        # Drop any rows that have non-numeric values in the columns we want to plot
        filtered_df.dropna(subset=[x_col_name, y_col_name], inplace=True)

        if filtered_df.empty:
            print("No valid data for plotting after cleaning non-numeric values.")
            return

        print(f"Plotting '{y_col_name}' vs. '{x_col_name}'.")

        # Create the plot
        plt.figure(figsize=(10, 7))
        plt.scatter(filtered_df[x_col_name], filtered_df[y_col_name], alpha=0.7)
        
        # Add labels and title
        plt.xlabel(f"Descriptor: {x_col_name} (eV)")
        plt.ylabel(f"Activity: {y_col_name} (V)")
        plt.title("Volcano Plot")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save the figure
        output_filename = 'volcano_plot.png'
        plt.savefig(output_filename)
        
        print(f"\nPlot successfully generated and saved as '{output_filename}' in the same directory.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except IndexError:
        print("Error: The file does not have enough columns. Please check the column indices.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Path to the data file
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d2.dat'
    
    filter_and_plot_data(DATA_FILE)