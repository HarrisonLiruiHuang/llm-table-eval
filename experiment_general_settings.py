# import necessary libraries
import os
import pandas as pd
import numpy as np
from collections import Counter
from litellm import batch_completion
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import random
import re
import time
import gc 

# Section underneath is where you can change experiment settings, feel free to modify!
#--------------------------------------------------------------------------------------------------------------

# Configurations
os.environ["OPENAI_API_KEY"] = "Enter your OpenAI API key here"
inquired_column = 'cdscode'
school_column = 'school'
county_column = 'county'
repeat_times = 100
sample_sizes = [100, 500, 1000, 2000, 3000, 5000]

# Uncomment settings to add settings to experiment 
settings = {
    'baseline': {
        'name': 'Baseline',
        'description': 'Just school, inquired column, county column'
    },
    # 'sorted_school': {
    #     'name': 'Sort School Column',
    #     'description': 'Same prompt but sort school column'
    # },
    # 'group_by_county': {
    #     'name': 'Group by County',
    #     'description': 'Same prompt but group by county column'
    # },
    # 'group_by_county_with_headers': {
    #     'name': 'Group by County with Headers',
    #     'description': 'Same prompt but group by county column with headers'
    # },
    # 'highlight_same_county': {
    #     'name': 'Highlight Same County',
    #     'description': 'Wrap rows with same county as target school with ** **'
    # },
    # 'group_sort_county_sort_school': {
    #     'name': 'Group Sort County Sort School',
    #     'description': 'Group by county, sort counties alphabetically, then sort schools within each county'
    # },
    # 'json_format': {
    #     'name': 'JSON Format',
    #     'description': 'Encode each column alongside each value in JSON-like format'
    # },
    # 'column_distance': {
    #     'name': 'Column Distance Test',
    #     'description': 'Test impact of distance between inquired column and school column'
    # },
    # 'remove_random_symbols': {
    #     'name': 'Remove Random Symbols',
    #     'description': 'Same as baseline but without random symbols and noise'
    # },
    # 'baseline_no_mention_symbols': {
    #     'name': 'Baseline No Mention Symbols',
    #     'description': 'Same dataset as baseline but prompt does not mention random symbols'
    # },
    # 'predicate_count_10': {
    #     'name': 'Predicate Count 10',
    #     'description': 'Add 10 additional predicates about the target row'
    # },
    # 'predicate_count_25': {
    #     'name': 'Predicate Count 25',
    #     'description': 'Add 25 additional predicates about the target row'
    # },
    # 'predicate_count_49': {
    #     'name': 'Predicate Count 49',
    #     'description': 'Add 49 additional predicates about the target row'
    # },
    # 'remove_nan_rows': {
    #     'name': 'Remove NaN Rows',
    #     'description': 'Same as baseline but remove all rows with NaN values from dataset'
    # },
    # 'baseline_mini': {
    #     'name': 'Baseline Mini',
    #     'description': 'Same as baseline but using GPT-4.1-mini model'
    # },
    # 'baseline_nano': {
    #     'name': 'Baseline Nano',
    #     'description': 'Same as baseline but using GPT-4.1-nano model'
    # }
}

#--------------------------------------------------------------------------------------------------------------


# Load data
import os
csv_paths = [
    'california_schools.csv',           # If running from root directory
    '../california_schools.csv',       # If running from experiments directory
    'experiments/../california_schools.csv'  # Alternative path
]

california_schools = None
for csv_path in csv_paths:
    if os.path.exists(csv_path):
        california_schools = pd.read_csv(csv_path, low_memory=False)
        print(f"📁 Loaded data from: {csv_path}")
        break

if california_schools is None:
    raise FileNotFoundError("Could not find california_schools.csv in any of the expected locations")

# Get unique county names from the dataset for cleaning
unique_counties = california_schools[county_column].unique().tolist()

# Clean function
def clean_response(response):
    response = response.strip()
    
    # Remove common prefixes
    prefixes = [
        "cdscode:", 
        "The value of inquired_column is:", 
        "The inquired_column is:", 
        "The cdscode is:"
    ]
    for prefix in prefixes:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
    
    # Remove county names that actually exist in the dataset
    for county in unique_counties:
        response = response.replace(county, "").replace(county.lower(), "")
    
    # Clean up any extra spaces or punctuation that might be left
    response = response.strip()
    response = response.replace("  ", " ")  # Remove double spaces
    response = response.strip(":")  # Remove trailing colons
    
    return response

# Function to create dataset string based on setting
# Random symbols to add to each field
symbols = ['@', '#', '$', '%', '&', '*', '!', '^', '~', '`', '+', '=', '|', '\\', '/', '?', '>', '<', '[', ']', '{', '}', '(', ')', '-', '_', ':', ';', '"', "'"]

def add_random_symbols(text, row_hash):
    """Add deterministic symbols and noise to text based on row hash"""
    # Use row_hash to generate deterministic noise
    hash_seed = hash(row_hash) % (2**32)  # Convert to 32-bit integer
    random.seed(hash_seed)
    
    symbol = random.choice(symbols)
    
    # Add deterministic number noise (1-3 random numbers)
    num_noise_count = random.randint(1, 3)
    noise_numbers = ''.join([str(random.randint(0, 9)) for _ in range(num_noise_count)])
    
    # Deterministic placement: before, after, or both
    placement = random.choice(['before', 'after', 'both'])
    
    if placement == 'before':
        return f"{noise_numbers}{symbol}{text}{symbol}"
    elif placement == 'after':
        return f"{symbol}{text}{symbol}{noise_numbers}"
    else: # both
        return f"{noise_numbers}{symbol}{text}{symbol}{noise_numbers}"

def create_dataset_string(sample, setting_key, target_county=None):
    if setting_key == 'baseline':
        # Header format: column names at beginning, then just values
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key == 'sorted_school':
        # Sort by school column
        sorted_sample = sample.sort_values(by=school_column)
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sorted_sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key == 'group_by_county':
        # Group by county, then encode same as baseline
        grouped_data = []
        header = "school, cdscode, county"
        for county, group in sample.groupby(county_column):
            for _, row in group.iterrows():
                school_seed = f"{row.name}_{row['school']}"
                cdscode_seed = f"{row.name}_{row[inquired_column]}"
                county_seed = f"{row.name}_{row[county_column]}"
                row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
                grouped_data.append(row_string)
        return f"{header}; " + '; '.join(grouped_data)
    
    elif setting_key == 'group_by_county_with_headers':
        # Group by county with headers
        grouped_data = []
        header = "school, cdscode, county"
        for county, group in sample.groupby(county_column):
            # Add header for this county and highlight if it's the target county
            county_seed = f"county_{county}"
            county_header = f"Schools in{add_random_symbols(str(county), county_seed)}:"
            if county == target_county:
                county_header = f"**{county_header}**"
            grouped_data.append(county_header)
            # Add the school data
            for _, row in group.iterrows():
                school_seed = f"{row.name}_{row['school']}"
                cdscode_seed = f"{row.name}_{row[inquired_column]}"
                county_seed = f"{row.name}_{row[county_column]}"
                row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
                grouped_data.append(row_string)
        return f"{header}; " + '; '.join(grouped_data)
    
    elif setting_key == 'highlight_same_county':
        # Highlight rows with same county as target school
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            school_entry = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            if target_county and row[county_column] == target_county:
                school_entry = f"**{school_entry}**"
            row_strings.append(school_entry)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key == 'group_sort_county_sort_school':
        # Group by county, sort counties alphabetically, then sort schools within each county
        grouped_data = []
        header = "school, cdscode, county"
        # Get all counties and sort them alphabetically
        counties = sorted(sample[county_column].unique())
        for county in counties:
            group = sample[sample[county_column] == county]
            # Sort schools within this county group alphabetically
            sorted_group = group.sort_values(by=school_column)
            for _, row in sorted_group.iterrows():
                school_seed = f"{row.name}_{row['school']}"
                cdscode_seed = f"{row.name}_{row[inquired_column]}"
                county_seed = f"{row.name}_{row[county_column]}"
                row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
                grouped_data.append(row_string)
        return f"{header}; " + '; '.join(grouped_data)
    
    elif setting_key == 'json_format':
        # JSON-like format: school: <name>, cdscode: <value>, county: <county>; Schools in <County2>: school: <name>, cdscode: <value>, county: <county>; ...
        # Group by county first, but list column names at beginning  
        header = "school, cdscode, county"
        grouped_data = []
        
        for county, group in sample.groupby(county_column):
            # Add county header with label
            county_header = f"Schools in {str(county)}:"
            if county == target_county:
                county_header = f"**{county_header}**"
            
            # Create JSON-like entries for each school in this county with noise
            county_entries = []
            for _, row in group.iterrows():
                school_seed = f"{row.name}_{row['school']}"
                cdscode_seed = f"{row.name}_{row[inquired_column]}"
                county_seed = f"{row.name}_{row[county_column]}"
                json_entry = f"school: {add_random_symbols(str(row['school']), school_seed)}, cdscode: {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, county: {add_random_symbols(str(row[county_column]), county_seed)}"
                county_entries.append(json_entry)
            
            # Combine county header with its entries
            if county_entries:
                grouped_data.append(county_header + " " + '; '.join(county_entries))
        
        return f"{header}; " + '; '.join(grouped_data)
    
    elif setting_key == 'column_distance':
        # Test different distances between inquired column and school column
        # Column positions to test: [(0,1), (0, 25), (0, 49)]
        # For now, use baseline format - the column positioning will be handled in the experiment loop
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key == 'remove_random_symbols':
        # Same as baseline but without random symbols and noise - clean data
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            # Use clean values without any symbols or noise
            row_string = f"{str(row['school'])}, {str(row[inquired_column])}, {str(row[county_column])}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key == 'baseline_no_mention_symbols':
        # Same dataset as baseline (with symbols) but prompt won't mention the symbols
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key in ['predicate_count_10', 'predicate_count_25', 'predicate_count_49']:
        # Extract the number of predicates from the setting key
        num_predicates = int(setting_key.split('_')[-1])
        
        # Same as baseline but with additional predicates about the target row
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key == 'remove_nan_rows':
        # Same as baseline but remove all rows with NaN values
        # Filter out rows that have NaN values in any column
        clean_sample = sample.dropna()
        
        header = "school, cdscode, county"
        row_strings = []
        for _, row in clean_sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    elif setting_key in ['baseline_mini', 'baseline_nano']:
        # Same dataset as baseline (with symbols) but using different models
        header = "school, cdscode, county"
        row_strings = []
        for _, row in sample.iterrows():
            school_seed = f"{row.name}_{row['school']}"
            cdscode_seed = f"{row.name}_{row[inquired_column]}"
            county_seed = f"{row.name}_{row[county_column]}"
            row_string = f"{add_random_symbols(str(row['school']), school_seed)}, {add_random_symbols(str(row[inquired_column]), cdscode_seed)}, {add_random_symbols(str(row[county_column]), county_seed)}"
            row_strings.append(row_string)
        return f"{header}; " + '; '.join(row_strings)
    
    return ''

def generate_additional_predicates(target_row, num_predicates):
    """Generate additional predicates about the target row using available columns"""
    # Available columns for creating predicates (excluding the main ones we already use)
    predicate_columns = [
        'street', 'streetabr', 'city', 'zip', 'state', 'phone', 'website', 
        'district', 'charter', 'fundingtype', 'doctype', 'soctype', 
        'edopsname', 'eilname', 'gsoffered', 'gsserved', 'virtual', 'magnet',
        'latitude', 'longitude', 'mailcity', 'mailzip', 'mailstate',
        'statustype', 'opendate', 'closeddate'
    ]
    
    predicates = []
    available_columns = [col for col in predicate_columns if col in target_row.index and pd.notna(target_row[col]) and str(target_row[col]).strip() != '']
    
    # Shuffle to get random order
    random.shuffle(available_columns)
    
    for i, col in enumerate(available_columns[:num_predicates]):
        value = str(target_row[col]).strip()
        if value and value.lower() != 'null' and value != '':
            # Create different types of predicates
            if col in ['latitude', 'longitude']:
                predicates.append(f"The school's {col} is {value}")
            elif col in ['charter', 'virtual', 'magnet']:
                predicates.append(f"The school is {'a ' + col if value == '1' else 'not a ' + col} school")
            elif col in ['phone', 'website']:
                predicates.append(f"The school's {col} is {value}")
            elif col in ['street', 'streetabr', 'city', 'mailcity']:
                predicates.append(f"The school's {col.replace('mail', 'mailing ')} is {value}")
            elif col in ['zip', 'mailzip']:
                predicates.append(f"The school's {col.replace('mail', 'mailing ')} code is {value}")
            elif col in ['opendate', 'closeddate']:
                predicates.append(f"The school's {col.replace('date', ' date')} is {value}")
            else:
                predicates.append(f"The school's {col} is {value}")
    
    return predicates

def create_dataset_with_column_distance(sample, school_pos, inquired_pos):
    """Create dataset string with specific column positions for school and inquired columns"""
    # Determine the maximum position needed
    max_pos = max(school_pos, inquired_pos)
    
    # Create column names - we'll fill with dummy columns
    columns = []
    for i in range(max_pos + 2):  # +2 to ensure we have enough columns including county
        if i == school_pos:
            columns.append("school")
        elif i == inquired_pos:
            columns.append("cdscode")
        else:
            columns.append(f"col_{i}")
    
    # Add county column at the end
    if "county" not in columns:
        columns.append("county")
        county_pos = len(columns) - 1
    else:
        county_pos = columns.index("county")
    
    # Create header [[memory:6789628]]
    header = ", ".join(columns)
    
    # Create rows with positioned data
    row_strings = []
    for _, row in sample.iterrows():
        # Initialize row data with dummy values
        row_data = []
        for i, col_name in enumerate(columns):
            if i == school_pos:
                school_seed = f"{row.name}_{row['school']}"
                row_data.append(add_random_symbols(str(row['school']), school_seed))
            elif i == inquired_pos:
                cdscode_seed = f"{row.name}_{row[inquired_column]}"
                row_data.append(add_random_symbols(str(row[inquired_column]), cdscode_seed))
            elif i == county_pos:
                county_seed = f"{row.name}_{row[county_column]}"
                row_data.append(add_random_symbols(str(row[county_column]), county_seed))
            else:
                # Add dummy data for filler columns
                dummy_seed = f"{row.name}_dummy_{i}"
                row_data.append(add_random_symbols(f"dummy{i}", dummy_seed))
        
        row_strings.append(", ".join(row_data))
    
    return f"{header}; " + '; '.join(row_strings)

# Parse multi-column response function
def parse_multi_column_response(response):
    """Parse response in format: school: [name], cdscode: [number], county: [name]"""
    try:
        result = {}
        
        # Handle multi-line responses (split by newlines)
        if '\n' in response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key in ['school', 'cdscode', 'county']:
                        result[key] = value
        
        # Handle comma-separated responses
        elif ',' in response:
            parts = response.split(',')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key in ['school', 'cdscode', 'county']:
                        result[key] = value
        
        # Handle single-line responses with colons
        else:
            if ':' in response:
                # Try to extract all three fields from a single line
                # Look for patterns like "school: X cdscode: Y county: Z"
                import re
                patterns = [
                    r'school:\s*([^,\n]+?)\s*cdscode:\s*([^,\n]+?)\s*county:\s*([^,\n]+)',
                    r'school:\s*([^,\n]+?)\s*county:\s*([^,\n]+?)\s*cdscode:\s*([^,\n]+)',
                    r'cdscode:\s*([^,\n]+?)\s*school:\s*([^,\n]+?)\s*county:\s*([^,\n]+)',
                    r'cdscode:\s*([^,\n]+?)\s*county:\s*([^,\n]+?)\s*school:\s*([^,\n]+)',
                    r'county:\s*([^,\n]+?)\s*school:\s*([^,\n]+?)\s*cdscode:\s*([^,\n]+)',
                    r'county:\s*([^,\n]+?)\s*cdscode:\s*([^,\n]+?)\s*school:\s*([^,\n]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        # Determine the order based on the pattern
                        if 'school:' in pattern and 'cdscode:' in pattern and 'county:' in pattern:
                            if pattern.startswith('school:'):
                                result['school'] = match.group(1).strip()
                                result['cdscode'] = match.group(2).strip()
                                result['county'] = match.group(3).strip()
                            elif pattern.startswith('cdscode:'):
                                result['cdscode'] = match.group(1).strip()
                                result['school'] = match.group(2).strip()
                                result['county'] = match.group(3).strip()
                            elif pattern.startswith('county:'):
                                result['county'] = match.group(1).strip()
                                result['school'] = match.group(2).strip()
                                result['cdscode'] = match.group(3).strip()
                        break
        
        # Check if we have all required fields
        if len(result) == 3 and all(key in result for key in ['school', 'cdscode', 'county']):
            return result
        else:
            return None
            
    except Exception as e:
        return None

# Results storage
combined_results = []

# Create results folder with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = f"results_{timestamp}"
os.makedirs(results_folder, exist_ok=True)
print(f"📁 Results will be saved to: {results_folder}")

# Create CSV file and write header
csv_path = os.path.join(results_folder, f"general_settings_results_{timestamp}.csv")
csv_columns = ["index", "accuracy", "setting", "setting_name", "sample_size", "repeat", "api_errors", "prediction_errors", "school_errors", "cdscode_errors", "county_errors", "total_errors"]
with open(csv_path, 'w', newline='') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(csv_columns)
print(f"📝 Created CSV file: {csv_path}")

# Function to generate live graph updates
def generate_live_graph(csv_path, results_folder, timestamp):
    """Generate and save an updated graph after each new data point"""
    try:
        # Clear any existing plots first
        plt.close('all')
        plt.clf()
        
        # Read current CSV data
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return
        
        # Create the graph
        plt.figure(figsize=(12, 8))
        
        # Create subplot for accuracy vs sample size
        plt.subplot(2, 1, 1)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#ff7f0e']
        
        # Get all unique settings from the data
        unique_settings = df["setting"].unique()
        for i, setting_key in enumerate(unique_settings):
            subset = df[df["setting"] == setting_key]
            if not subset.empty:
                # Get the setting name from the data
                setting_name = subset["setting_name"].iloc[0]
                means = subset.groupby("sample_size")["accuracy"].mean()
                plt.plot(means.index, means.values, marker='o', label=setting_name, 
                         color=colors[i % len(colors)], linewidth=2, markersize=8)
        
        plt.xlabel("Sample Size")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Live Accuracy vs Sample Size - {timestamp}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create subplot for detailed comparison
        plt.subplot(2, 1, 2)
        sample_sizes_in_data = sorted(df['sample_size'].unique())
        x_pos = np.arange(len(sample_sizes_in_data))
        num_settings = len(unique_settings)
        width = 0.8 / num_settings if num_settings > 0 else 0.12  # Dynamic width based on number of settings
        
        for i, setting_key in enumerate(unique_settings):
            subset = df[df["setting"] == setting_key]
            if not subset.empty:
                setting_name = subset["setting_name"].iloc[0]
                means = subset.groupby("sample_size")["accuracy"].mean()
                # Align with available sample sizes
                bar_values = [means.get(size, 0) for size in sample_sizes_in_data]
                plt.bar(x_pos + i*width, bar_values, width, label=setting_name, 
                        color=colors[i % len(colors)], alpha=0.8)
        
        plt.xlabel("Sample Size")
        plt.ylabel("Accuracy (%)")
        plt.title("Live Detailed Accuracy Comparison")
        plt.xticks(x_pos + (num_settings-1)*width/2, sample_sizes_in_data)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the live graph
        live_png_path = os.path.join(results_folder, f"live_general_settings_comparison_{timestamp}.png")
        plt.savefig(live_png_path, dpi=300, bbox_inches='tight')
        plt.close('all')  # Close all figures to free memory
        plt.clf()  # Clear the current figure
        gc.collect()  # Force garbage collection
        
        print(f"📊 Generated live graph: {live_png_path}")
        
    except Exception as e:
        print(f"⚠️ Error generating live graph: {e}")
        # Ensure cleanup even on error
        try:
            plt.close('all')
            plt.clf()
            gc.collect()
        except:
            pass

# At the top of your file, add this mapping
setting_to_number = {
    'baseline': 1,
    'sorted_school': 2, 
    'group_by_county': 3,
    'group_by_county_with_headers': 4,
    'highlight_same_county': 5,
    'group_sort_county_sort_school': 6,
    'json_format': 7,
    'column_distance_0_1': 8,
    'column_distance_0_25': 9,
    'column_distance_0_49': 10,
    'remove_random_symbols': 11,
    'baseline_no_mention_symbols': 12,
    'predicate_count_10': 13,
    'predicate_count_25': 14,
    'predicate_count_49': 15,
    'remove_nan_rows': 16,
    'baseline_mini': 17,
    'baseline_nano': 18
}

# Restructured experiment loop to batch across multiple repetitions
for sample_size in sample_sizes:
    print(f"\n=== Processing sample_size={sample_size} ===")
    
    # Collect ALL messages for ALL settings and ALL repetitions
    all_messages = []
    record_log = []
    
    for repeat in range(repeat_times):
        print(f"  Preparing repeat {repeat + 1}/{repeat_times}...")
        
        # Draw ONE sample for this (sample_size, repeat) combination
        # This sample will be reused for ALL settings to ensure fair comparison
        sample_seed = sample_size * 1000 + repeat
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        
        sample = (
            california_schools
            .drop_duplicates(subset='school')
            .sample(n=sample_size, random_state=sample_seed)
            .reset_index(drop=True)
        )[['school', inquired_column, county_column]].copy()

        # Use middle index for testing
        index_list = [sample_size//2]
        
        for setting_key, setting_info in settings.items():
            # Handle column distance test separately with multiple position pairs
            if setting_key == 'column_distance':
                column_positions = [(0,1), (0, 25), (0, 49)]
                for school_pos, inquired_pos in column_positions:
                    for idx in index_list:
                        school_name = sample.at[idx, 'school']
                        county_name = sample.at[idx, 'county']
                        expected = str(sample.at[idx, inquired_column])
                        target_county = sample.at[idx, county_column]

                        # Create dataset string with specific column positions
                        labeled_string_df = create_dataset_with_column_distance(sample, school_pos, inquired_pos)
                        
                        # Create prompt for column distance test
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data has school, cdscode, and county columns positioned at different distances from each other, with dummy columns in between.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""

                        all_messages.append([{"role": "user", "content": prompt}])
                        record_log.append({
                            "expected": expected,
                            "index": idx,
                            "school": school_name,
                            "target_county": target_county,
                            "repeat": repeat,
                            "setting": f"{setting_key}_{school_pos}_{inquired_pos}",
                            "setting_name": f"{setting_info['name']} (Pos: {school_pos},{inquired_pos})",
                            "sample_size": sample_size,
                            "sample_seed": sample_seed
                        })
            else:
                for idx in index_list:
                    school_name = sample.at[idx, 'school']
                    county_name = sample.at[idx, 'county']
                    expected = str(sample.at[idx, inquired_column])
                    target_county = sample.at[idx, county_column]

                    # Create dataset string based on setting (always pass target_county)
                    labeled_string_df = create_dataset_string(sample, setting_key, target_county)
                
                    # Create setting-specific prompt that explains the data formatting
                    if setting_key in ['baseline', 'baseline_mini', 'baseline_nano']:
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is presented in the original order without any special formatting.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'sorted_school':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is sorted alphabetically by school name.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'group_by_county':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is grouped by county, with schools from the same county appearing together.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'group_by_county_with_headers':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is grouped by county with clear county headers. The target school's county header is highlighted with ** **.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'highlight_same_county':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is in original order, but schools in the same county as the target school are highlighted with ** **.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'group_sort_county_sort_school':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is grouped by county (counties sorted alphabetically), then schools within each county are sorted alphabetically.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'json_format':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is in JSON-like format where each column name is paired with its value. Schools are grouped by county with county headers. The target school's county header is highlighted with ** **.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'remove_random_symbols':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is presented in clean format without any random symbols or noise.
                        Note: The data is in the original order without any special formatting.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'baseline_no_mention_symbols':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is presented in the original order without any special formatting.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key in ['predicate_count_10', 'predicate_count_25', 'predicate_count_49']:
                        # Get the number of predicates and generate them for the target row
                        num_predicates = int(setting_key.split('_')[-1])
                        target_row = sample.iloc[idx]
                        predicates = generate_additional_predicates(target_row, num_predicates)
                        predicates_text = "\n".join([f"- {pred}" for pred in predicates])
                        
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is presented in the original order without any special formatting.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        
                        Additional information about the target school:
{predicates_text}
                        
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    elif setting_key == 'remove_nan_rows':
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: Data is presented in the original order without any special formatting.
                        Note: All rows with missing or incomplete data have been removed from the dataset.
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""
                    else:
                        prompt = f"""Dataset: {labeled_string_df}
                        Note: The data contains random symbols (like @, #, $, %, etc.) and random number noise around the actual values.
                        Note: You must determine the correct data format and extract the clean values by removing all noise.
                        Target school: {school_name} (County: {target_county})
                        Find the school name, cdscode, and county for: {school_name}
                        Respond in format: school: [name], cdscode: [number], county: [name]"""

                    all_messages.append([{"role": "user", "content": prompt}])
                    record_log.append({
                        "expected": expected,
                        "index": idx,
                        "school": school_name,
                        "target_county": target_county,
                        "repeat": repeat,
                        "setting": setting_key,
                        "setting_name": setting_info['name'],
                        "sample_size": sample_size,
                        "sample_seed": sample_seed
                    })

    print(f"  📝 Prepared {len(all_messages)} messages for all settings and all repetitions")
    
    # Process ALL messages in batches
    print(f"  🚀 Processing {len(all_messages)} messages in batches...")
    
    if sample_size >= 8000:
        print(f"  🔄 Large sample size ({sample_size}) detected - using rate limit handling...")
        batch_size = 32
    else:
        batch_size = 64  # Larger batches for smaller sample sizes
    
    responses = []
    
    # Group messages by model type to process them separately
    model_batches = {}
    for i, (message, record) in enumerate(zip(all_messages, record_log)):
        setting = record["setting"]
        if setting == 'baseline_mini':
            model_name = "gpt-4.1-mini"
        elif setting == 'baseline_nano':
            model_name = "gpt-4.1-nano"
        else:
            model_name = "gpt-4.1"
        
        if model_name not in model_batches:
            model_batches[model_name] = []
        model_batches[model_name].append((i, message, record))
    
    # Process each model's messages separately
    responses = [None] * len(all_messages)  # Pre-allocate response array
    
    for model_name, model_messages in model_batches.items():
        print(f"  🤖 Processing {len(model_messages)} messages with {model_name}...")
        messages_only = [msg for _, msg, _ in model_messages]
        
        for i in range(0, len(messages_only), batch_size):
            batch = messages_only[i:i + batch_size]
            batch_indices = [idx for idx, _, _ in model_messages[i:i + batch_size]]
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    batch_responses = batch_completion(
                        model=model_name,
                        messages=batch,
                        temperature=0
                    )
                    
                    # Put responses back in correct positions
                    for j, response in enumerate(batch_responses):
                        responses[batch_indices[j]] = response
                    
                    print(f"    {model_name} Batch {i//batch_size + 1}/{(len(messages_only)-1)//batch_size + 1} completed ({len(batch)} messages)")
                    
                    if i + batch_size < len(messages_only):
                        print(f"    Sleeping 2 seconds between batches...")
                        time.sleep(2)
                    
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Rate limit" in error_msg or "TPM" in error_msg:
                        import re
                        wait_match = re.search(r'(\d+\.?\d*)s', error_msg)
                        wait_time = float(wait_match.group(1)) if wait_match else 5.0
                        
                        print(f"    Rate limit hit, waiting {wait_time + 2} seconds... (attempt {retry_count + 1}/{max_retries})")
                        time.sleep(wait_time + 2)
                        retry_count += 1
                    else:
                        print(f"    Unexpected error: {e}")
                        retry_count += 1
                        time.sleep(2)
            
            if retry_count >= max_retries:
                print(f"    Failed to process {model_name} batch {i//batch_size + 1} after {max_retries} retries")
                for j in range(len(batch)):
                    responses[batch_indices[j]] = Exception(f"Batch failed after {max_retries} retries")
    
    print(f"  ✅ Completed {len(all_messages)} messages in {len(all_messages)//batch_size + 1} batches")

    # Process responses for ALL settings and ALL repetitions
    print(f"  📊 Processing responses for all settings and repetitions...")
    
    # Group responses by setting
    setting_responses = {}
    for i, (res, record) in enumerate(zip(responses, record_log)):
        setting_key = record["setting"]
        if setting_key not in setting_responses:
            setting_responses[setting_key] = []
        setting_responses[setting_key].append((res, record))

    # Process each setting's responses - handle column_distance sub-settings
    all_setting_keys = []
    for setting_key, setting_info in settings.items():
        if setting_key == 'column_distance':
            # Add all column distance sub-settings
            all_setting_keys.extend(['column_distance_0_1', 'column_distance_0_25', 'column_distance_0_49'])
        else:
            all_setting_keys.append(setting_key)
    
    for actual_setting_key in all_setting_keys:
        # Get the display name for this setting
        if actual_setting_key.startswith('column_distance_'):
            base_setting_info = settings['column_distance']
            positions = actual_setting_key.replace('column_distance_', '').replace('_', ',')
            display_name = f"{base_setting_info['name']} (Pos: {positions})"
        else:
            display_name = settings[actual_setting_key]['name']
            
        print(f"    Processing {display_name}...")
        
        if actual_setting_key not in setting_responses:
            print(f"    ⚠️ No responses found for {display_name}")
            continue
            
        setting_data = setting_responses[actual_setting_key]
        
        # Group by repeat to calculate accuracy per repeat
        repeat_results = {}
        for res, record in setting_data:
            repeat = record["repeat"]
            if repeat not in repeat_results:
                repeat_results[repeat] = []
            repeat_results[repeat].append((res, record))
        
        # Calculate accuracy for each repeat
        for repeat, repeat_data in repeat_results.items():
            correct_counter = Counter()
            total_counter = Counter()
            
            api_errors = 0
            prediction_errors = 0
            school_errors = 0
            cdscode_errors = 0
            county_errors = 0

            for res, record in repeat_data:
                if isinstance(res, Exception):
                    print(f"      Error in repeat {repeat}: {res}")
                    api_errors += 1
                    continue

                prediction = res["choices"][0]["message"]["content"].strip()
                expected_school = record["school"]
                expected_cdscode = record["expected"]
                expected_county = record["target_county"]
                idx = record["index"]

                total_counter[idx] += 1
                
                parsed_response = parse_multi_column_response(prediction)
                
                if parsed_response:
                    expected_school_str = str(expected_school) if pd.notna(expected_school) else ''
                    expected_cdscode_str = str(expected_cdscode) if pd.notna(expected_cdscode) else ''
                    expected_county_str = str(expected_county) if pd.notna(expected_county) else ''
                    
                    school_correct = parsed_response.get('school', '').strip().lower() == expected_school_str.strip().lower()
                    cdscode_correct = parsed_response.get('cdscode', '').strip() == expected_cdscode_str.strip()
                    county_correct = parsed_response.get('county', '').strip().lower() == expected_county_str.strip().lower()
                    
                    if school_correct and cdscode_correct and county_correct:
                        correct_counter[idx] += 1
                    else:
                        if not school_correct: school_errors += 1
                        if not cdscode_correct: cdscode_errors += 1
                        if not county_correct: county_errors += 1
                        
                        print(f"      Mismatch in repeat {repeat}, index {idx}, setting={record['setting']}")
                        print(f"      Prediction: {prediction}")
                        print(f"      Expected: school: {expected_school_str}, cdscode: {expected_cdscode_str}, county: {expected_county_str}")
                        print(f"      Got: school: {parsed_response.get('school', 'N/A')}, cdscode: {parsed_response.get('cdscode', 'N/A')}, county: {parsed_response.get('county', 'N/A')}")
                        print("      ---")
                        prediction_errors += 1
                else:
                    prediction_errors += 1
                    print(f"      Could not parse response in repeat {repeat}, index {idx}: {prediction}")

            # Store result for this setting and repeat
            result_data = [
                {
                    "index": idx,
                    "accuracy": correct_counter[idx] / total_counter[idx] * 100 if total_counter[idx] else 0,
                    "setting": actual_setting_key,
                    "setting_name": display_name,
                    "sample_size": sample_size,
                    "repeat": repeat,
                    "api_errors": api_errors,
                    "prediction_errors": prediction_errors,
                    "school_errors": school_errors,
                    "cdscode_errors": cdscode_errors,
                    "county_errors": county_errors,
                    "total_errors": api_errors + prediction_errors
                }
                for idx in index_list
            ]
            combined_results.append(pd.DataFrame(result_data))
            
            # Save results incrementally to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for result in result_data:
                    writer.writerow([
                        result["index"],
                        result["accuracy"],
                        result["setting"],
                        result["setting_name"],
                        result["sample_size"],
                        result["repeat"],
                        result["api_errors"],
                        result["prediction_errors"],
                        result["school_errors"],
                        result["cdscode_errors"],
                        result["county_errors"],
                        result["total_errors"]
                    ])
            
            print(f"      💾 Saved results for {display_name}, repeat {repeat}")
            
            # Print error summary for this setting and repeat
            print(f"      Error summary for {display_name}, repeat {repeat}:")
            print(f"        API errors: {api_errors}")
            print(f"        Prediction errors: {prediction_errors}")
            print(f"        School errors: {school_errors}")
            print(f"        CDSCode errors: {cdscode_errors}")
            print(f"        County errors: {county_errors}")
            print(f"        Total errors: {api_errors + prediction_errors}")

    # Generate updated graph after processing all settings and repetitions for this sample size
    generate_live_graph(csv_path, results_folder, timestamp)
    
    # Clean up memory after each sample size
    gc.collect()

# Combine results
final_result_df = pd.concat(combined_results, ignore_index=True)

# Avoid duplicate rows
final_result_df = final_result_df.drop_duplicates(subset=["setting", "sample_size", "index"], keep="last")

# Output average
print("\n=== Average Accuracy by sample_size and setting ===")
accuracy_summary = final_result_df.groupby(["sample_size", "setting_name"])["accuracy"].mean()
print(accuracy_summary)

# Print detailed results
for setting_key, setting_info in settings.items():
    print(f"\n{setting_info['name']}:")
    for sample_size in sample_sizes:
        subset = final_result_df[(final_result_df["sample_size"] == sample_size) & (final_result_df["setting"] == setting_key)]
        if not subset.empty:
            mean_accuracy = subset["accuracy"].mean()
            print(f"  Sample Size {sample_size}: {mean_accuracy:.2f}%")

# Final graph is already generated live, just show the final version
print(f"📊 Final graph available at: {os.path.join(results_folder, f'live_general_settings_comparison_{timestamp}.png')}")
png_path = os.path.join(results_folder, f"live_general_settings_comparison_{timestamp}.png")

# Results already saved incrementally, just confirm
print(f"✅ Results already saved incrementally to {csv_path}")
print(f"✅ Graph saved to {png_path}")

# Print summary statistics
print("\n=== Summary Statistics ===")
summary_stats = final_result_df.groupby(['setting_name', 'sample_size'])['accuracy'].agg(['mean', 'std']).round(2)
print(summary_stats)

# Print error summary
print("\n=== Error Summary ===")
error_summary = final_result_df.groupby(['setting_name', 'sample_size'])[['api_errors', 'prediction_errors', 'total_errors']].sum()
print(error_summary)

# Print total errors across all settings
print("\n=== Total Errors Across All Settings ===")
total_api = final_result_df['api_errors'].sum()
total_prediction = final_result_df['prediction_errors'].sum()
total_all = final_result_df['total_errors'].sum()
print(f"Total API errors: {total_api}")
print(f"Total prediction errors: {total_prediction}")
print(f"Total all errors: {total_all}")

# Save summary to results folder
summary_path = os.path.join(results_folder, f"experiment_summary_{timestamp}.txt")
with open(summary_path, 'w') as f:
    f.write(f"Experiment Summary - {timestamp}\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Settings Tested:\n")
    for setting_key, setting_info in settings.items():
        f.write(f"- {setting_info['name']}: {setting_info['description']}\n")
    
    f.write(f"\nSample Sizes: {sample_sizes}\n")
    f.write(f"Repeat Times: {repeat_times}\n")
    f.write(f"Total Experiments: {len(final_result_df)}\n\n")
    
    f.write("Accuracy Summary:\n")
    f.write(str(accuracy_summary) + "\n\n")
    
    f.write("Error Summary:\n")
    f.write(str(error_summary) + "\n\n")
    
    f.write(f"Total API Errors: {total_api}\n")
    f.write(f"Total Prediction Errors: {total_prediction}\n")
    f.write(f"Total All Errors: {total_all}\n")

print(f"📄 Summary saved to {summary_path}")
print(f"🎯 All results saved in: {results_folder}")