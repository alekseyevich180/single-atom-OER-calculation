import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_loader import filter_and_load_data

def create_correlation_plot(df):
    """
    Creates and saves a scatter plot with two datasets and their trendlines.
    - X-axis: 'deltaE-OH' (column 7)
    - Y-axis 1: 'deltaE-OOH' (column 9)
    - Y-axis 2: 'deltaE-O' (column 8)
    """
    if df is None or df.empty:
        print("Input DataFrame is empty, cannot create plot.")
        return

    try:
        # Define column indices for clarity
        x_col_index = 6   # deltaE-OH
        y1_col_index = 8  # deltaE-OOH
        y2_col_index = 7  # deltaE-O

        # Get column names from the DataFrame
        x_col_name = df.columns[x_col_index]
        y1_col_name = df.columns[y1_col_index]
        y2_col_name = df.columns[y2_col_index]
        
        # Ensure all required columns are numeric, converting non-numeric values to NaN
        for col in [x_col_name, y1_col_name, y2_col_name]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create separate dataframes for each plot, dropping rows with missing values
        df_plot1 = df[[x_col_name, y1_col_name]].dropna()
        df_plot2 = df[[x_col_name, y2_col_name]].dropna()

        print(f"Plotting '{y1_col_name}' vs. '{x_col_name}' using {len(df_plot1)} data points.")
        print(f"Plotting '{y2_col_name}' vs. '{x_col_name}' using {len(df_plot2)} data points.")

        # --- Create the Plot ---
        plt.figure(figsize=(12, 8))

        # --- Dataset 1: deltaE-OOH vs deltaE-OH ---
        if not df_plot1.empty:
            x1_data, y1_data = df_plot1[x_col_name], df_plot1[y1_col_name]
            plt.scatter(x1_data, y1_data, alpha=0.6, label=f'{y1_col_name} vs. {x_col_name}')
            
            # Calculate and plot trendline
            m1, b1 = np.polyfit(x1_data, y1_data, 1)
            plt.plot(x1_data, m1*x1_data + b1, color='navy', linestyle='--', label=f'Trend for {y1_col_name}')

        # --- Dataset 2: deltaE-O vs deltaE-OH ---
        if not df_plot2.empty:
            x2_data, y2_data = df_plot2[x_col_name], df_plot2[y2_col_name]
            plt.scatter(x2_data, y2_data, alpha=0.6, marker='^', label=f'{y2_col_name} vs. {x_col_name}')
            
            # Calculate and plot trendline
            m2, b2 = np.polyfit(x2_data, y2_data, 1)
            plt.plot(x2_data, m2*x2_data + b2, color='darkgreen', linestyle='--', label=f'Trend for {y2_col_name}')

        # --- Final Touches ---
        plt.xlabel(f"Adsorption Energy: {x_col_name} (eV)")
        plt.ylabel("Adsorption Energy (eV)")
        plt.title("Correlation of Adsorption Energies")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        output_filename = 'adsorption_energy_correlation.png'
        plt.savefig(output_filename)
        
        print(f"\nPlot successfully generated and saved as '{output_filename}'.")

    except IndexError:
        print("Error: The DataFrame does not have the required columns. Please check the column indices.")
    except Exception as e:
        print(f"An unexpected error occurred during the plotting process: {e}")

if __name__ == '__main__':
    # Define the path to the data file
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d2.dat'
    
    # Load and filter the data using the imported function
    filtered_dataframe = filter_and_load_data(DATA_FILE)
    
    # Create the correlation plot
    create_correlation_plot(filtered_dataframe)