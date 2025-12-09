import pandas as pd
import matplotlib.pyplot as plt

def filter_and_plot_data(file_path):
    """
    Reads data from the specified file, filters it, and creates a volcano plot.
    
    - Filters for rows where the value in column 13 (index 12) is between 2.0 and 4.0.
    - Plots column 19 vs. column 22 (1-based indices) if those columns exist.
    """
    try:
        # Build column names to keep extra columns that appear after the header.
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_fields = lines[0].split()
        max_len = max(len(line.split()) for line in lines)
        column_names = header_fields + [f'extra_{i}' for i in range(len(header_fields), max_len)]

        # Use read_csv with whitespace delimiter and treat the first row as a header.
        # Skip the first data row (index 1) because subsequent rows have no header.
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=0,
            names=column_names,
            engine='python',
            skiprows=[1]
        )

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
            col_stats = pd.to_numeric(df[filter_column_name], errors='coerce')
            print(f"No data remains after filtering. Column '{filter_column_name}' range: "
                  f"min={col_stats.min()}, max={col_stats.max()}")
            return

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

        # --- 2. Plotting Step ---
        # Define column indices for plotting (0-based). 1-based indices: 19 and 22.
        x_col_index = 16
        y_col_index = 20

        if x_col_index >= len(filtered_df.columns) or y_col_index >= len(filtered_df.columns):
            print(f"Error: Requested plot columns (19, 22) exceed available columns ({len(filtered_df.columns)}).")
            print("Available columns (0-based):")
            for i, col in enumerate(filtered_df.columns):
                print(f"  {i}: {col}")
            return

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

        # Annotate by base element name (strip suffixes like _pv, _sv) with minimal overlap.
        # Place one label per base element at the mean coordinate of its points,
        # then stagger labels if they land on the same spot.
        grouped = filtered_df.assign(_base=filtered_df['element'].astype(str).str.split('_').str[0])
        label_positions = grouped.groupby('_base')[[x_col_name, y_col_name]].mean().reset_index()

        # Default: no offset. For specific crowded labels, apply small manual offsets.
        label_offsets = {
            'Ir': (10, 6),
            'Cr': (-10, 6),
            'Mn': (10, -6),
            'Pb': (-10, -6),
        }
        for _, row in label_positions.iterrows():
            label = row['_base']
            x_val, y_val = row[x_col_name], row[y_col_name]
            dx, dy = label_offsets.get(label, (0, 0))
            plt.annotate(
                label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8
            )
        
        # Add labels and title
        plt.xlabel(f"Descriptor: {x_col_name} (eV)")
        plt.ylabel(f"Activity: {y_col_name} (eV)")
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
    DATA_FILE = 'c:\\Users\\yingkaiwu\\Desktop\\single-atom\\volcano\\pbe-d3.dat'
    
    filter_and_plot_data(DATA_FILE)
