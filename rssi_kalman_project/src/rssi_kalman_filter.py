import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import warnings
import os
warnings.filterwarnings('ignore')

class RSSIKalmanFilter:
    def __init__(self, process_noise=1.0, measurement_noise=10.0):
        """
        Initialize RSSI Kalman Filter
        
        Args:
            process_noise: Process noise variance (how much the signal can change)
            measurement_noise: Measurement noise variance (RSSI measurement uncertainty)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.kf = None
        
    def setup_filter(self, initial_rssi):
        """Setup the Kalman filter with initial RSSI value"""
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State vector: [rssi, rssi_velocity]
        self.kf.x = np.array([[initial_rssi], [0.0]])  # Initial state
        
        # State transition matrix (constant velocity model)
        dt = 1.0  # Time step (can be adjusted based on sampling rate)
        self.kf.F = np.array([[1.0, dt],
                              [0.0, 1.0]])
        
        # Measurement function (we only measure RSSI, not velocity)
        self.kf.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        
        # Measurement noise covariance
        self.kf.R = np.array([[self.measurement_noise]])
        
        # Initial covariance matrix
        self.kf.P = np.eye(2) * 500.0  # High initial uncertainty
    
    def filter_rssi_sequence(self, rssi_values):
        """Filter a sequence of RSSI values"""
        if len(rssi_values) == 0:
            return []
        
        # Setup filter with first RSSI value
        self.setup_filter(rssi_values[0])
        
        filtered_values = []
        
        for rssi in rssi_values:
            # Predict step
            self.kf.predict()
            
            # Update step with measurement
            self.kf.update(rssi)
            
            # Store filtered value
            filtered_values.append(self.kf.x[0, 0])
        
        return filtered_values

def load_rssi_data(raw_file, filtered_file=None):
    """Load RSSI data from CSV files"""
    print(f"Attempting to load: {raw_file}")
    
    # Try different parsing methods
    raw_data = None
    parsing_methods = [
        {'sep': ','},  # Standard CSV
        {'sep': r'\s+'},  # Whitespace separated
        {'sep': '\t'},  # Tab separated
        {'sep': r'\s+', 'engine': 'python'},  # Python engine for regex
    ]
    
    for i, method in enumerate(parsing_methods):
        try:
            print(f"Trying parsing method {i+1}: {method}")
            raw_data = pd.read_csv(raw_file, **method)
            print(f"Successfully loaded with method {i+1}")
            print(f"Shape: {raw_data.shape}")
            print(f"Columns: {raw_data.columns.tolist()}")
            print(f"First few rows:")
            print(raw_data.head(3))
            
            # Check if we have the expected columns
            expected_cols = ['name', 'locationStatus', 'timestamp', 'rssiOne', 'rssiTwo']
            if all(col in raw_data.columns for col in expected_cols):
                print("✅ All expected columns found!")
                break
            else:
                print(f"❌ Missing columns. Expected: {expected_cols}")
                print(f"Found: {raw_data.columns.tolist()}")
                raw_data = None
                
        except Exception as e:
            print(f"Method {i+1} failed: {str(e)}")
            continue
    
    if raw_data is None:
        print("❌ All parsing methods failed. Let's try manual inspection...")
        # Read first few lines to inspect format
        try:
            with open(raw_file, 'r') as f:
                lines = [f.readline().strip() for _ in range(5)]
                print("First 5 lines of the file:")
                for i, line in enumerate(lines):
                    print(f"Line {i+1}: '{line}'")
        except Exception as e:
            print(f"Could not read file: {e}")
        
        raise ValueError(f"Could not parse {raw_file}. Please check the file format.")
    
    # Load filtered data if provided
    filtered_data = None
    if filtered_file and os.path.exists(filtered_file):
        print(f"\nLoading filtered data: {filtered_file}")
        try:
            # Use the same method that worked for raw data
            for method in parsing_methods:
                try:
                    filtered_data = pd.read_csv(filtered_file, **method)
                    print(f"Filtered data shape: {filtered_data.shape}")
                    break
                except:
                    continue
        except Exception as e:
            print(f"Warning: Could not load filtered data: {e}")
    
    return raw_data, filtered_data

def process_device_rssi(data, device_name, rssi_column='rssiOne'):
    """Process RSSI data for a specific device"""
    # Filter data for specific device
    device_data = data[data['name'] == device_name].copy()
    
    if len(device_data) == 0:
        print(f"No data found for device: {device_name}")
        return None, None
    
    # Sort by timestamp
    device_data = device_data.sort_values('timestamp')
    
    # Extract RSSI values and timestamps
    rssi_values = device_data[rssi_column].values
    timestamps = device_data['timestamp'].values
    
    return rssi_values, timestamps

def filter_all_devices_rssi(raw_data, process_noise=1.0, measurement_noise=10.0):
    """Filter RSSI for all devices in the dataset"""
    # Get unique device names
    devices = raw_data['name'].unique()
    
    results = {}
    
    for device in devices:
        print(f"Processing device: {device}")
        
        # Process both rssiOne and rssiTwo
        for rssi_col in ['rssiOne', 'rssiTwo']:
            rssi_values, timestamps = process_device_rssi(raw_data, device, rssi_col)
            
            if rssi_values is not None:
                # Create and apply Kalman filter
                kf = RSSIKalmanFilter(process_noise=process_noise, 
                                    measurement_noise=measurement_noise)
                filtered_rssi = kf.filter_rssi_sequence(rssi_values)
                
                # Store results
                key = f"{device}_{rssi_col}"
                results[key] = {
                    'raw': rssi_values,
                    'filtered': filtered_rssi,
                    'timestamps': timestamps,
                    'device': device,
                    'rssi_type': rssi_col
                }
    
    return results

def plot_comparison(results, device_key, comparison_data=None, save_plots=True, output_dir="output"):
    """Plot raw vs filtered RSSI for comparison"""
    if device_key not in results:
        print(f"Device key {device_key} not found in results")
        return
    
    data = results[device_key]
    
    plt.figure(figsize=(12, 8))
    
    # Plot raw RSSI
    plt.subplot(2, 1, 1)
    plt.plot(data['timestamps'], data['raw'], 'b-', alpha=0.7, label='Raw RSSI')
    plt.plot(data['timestamps'], data['filtered'], 'r-', linewidth=2, label='Kalman Filtered')
    
    # Track if we have reference data for validation
    has_reference = False
    rmse_kalman = None
    
    if comparison_data is not None:
        # Add ground truth filtered data if available
        device_name = data['device']
        rssi_col = data['rssi_type']
        comp_rssi, comp_timestamps = process_device_rssi(comparison_data, device_name, rssi_col)
        if comp_rssi is not None:
            plt.plot(comp_timestamps, comp_rssi, 'g--', linewidth=2, label='Reference Filtered')
            has_reference = True
            
            # Calculate RMSE between Kalman filter and reference
            if len(comp_rssi) == len(data['filtered']):
                rmse_kalman = np.sqrt(np.mean((np.array(data['filtered']) - comp_rssi)**2))
    
    plt.xlabel('Timestamp')
    plt.ylabel('RSSI (dBm)')
    plt.title(f'RSSI Filtering Comparison - {device_key}')
    if rmse_kalman is not None:
        plt.title(f'RSSI Filtering Comparison - {device_key}\nRMSE vs Reference: {rmse_kalman:.2f} dBm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = np.array(data['raw']) - np.array(data['filtered'])
    plt.plot(data['timestamps'], residuals, 'purple', alpha=0.7)
    plt.xlabel('Timestamp')
    plt.ylabel('Residuals (Raw - Filtered)')
    plt.title('Filtering Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot_filename = os.path.join(output_dir, f"rssi_comparison_{device_key.replace(':', '_')}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_filename}")
    
    plt.show()
    
    return rmse_kalman, has_reference

def evaluate_filter_performance(results, reference_data=None, save_results=True, output_dir="output"):
    """Evaluate filter performance with metrics"""
    print("\n=== Filter Performance Evaluation ===")
    
    # Store all results for saving
    evaluation_results = []
    rmse_scores = []
    
    for device_key, data in results.items():
        raw_rssi = np.array(data['raw'])
        filtered_rssi = np.array(data['filtered'])
        
        # Calculate basic metrics
        noise_reduction = np.std(raw_rssi) - np.std(filtered_rssi)
        smoothness = np.mean(np.abs(np.diff(filtered_rssi)))
        raw_smoothness = np.mean(np.abs(np.diff(raw_rssi)))
        smoothness_improvement = raw_smoothness - smoothness
        
        # Calculate RMSE against reference if available
        rmse_vs_reference = None
        correlation_with_reference = None
        
        if reference_data is not None:
            device_name = data['device']
            rssi_col = data['rssi_type']
            ref_rssi, ref_timestamps = process_device_rssi(reference_data, device_name, rssi_col)
            
            if ref_rssi is not None and len(ref_rssi) == len(filtered_rssi):
                rmse_vs_reference = np.sqrt(np.mean((filtered_rssi - ref_rssi)**2))
                correlation_with_reference = np.corrcoef(filtered_rssi, ref_rssi)[0, 1]
                rmse_scores.append(rmse_vs_reference)
        
        # Store results
        result_dict = {
            'device': device_key,
            'raw_std': np.std(raw_rssi),
            'filtered_std': np.std(filtered_rssi),
            'noise_reduction': noise_reduction,
            'raw_smoothness': raw_smoothness,
            'filtered_smoothness': smoothness,
            'smoothness_improvement': smoothness_improvement,
            'rmse_vs_reference': rmse_vs_reference,
            'correlation_with_reference': correlation_with_reference
        }
        evaluation_results.append(result_dict)
        
        # Print results
        print(f"\nDevice: {device_key}")
        print(f"  Raw RSSI std: {np.std(raw_rssi):.2f} dBm")
        print(f"  Filtered RSSI std: {np.std(filtered_rssi):.2f} dBm")
        print(f"  Noise reduction: {noise_reduction:.2f} dBm")
        print(f"  Raw smoothness: {raw_smoothness:.2f}")
        print(f"  Filtered smoothness: {smoothness:.2f}")
        print(f"  Smoothness improvement: {smoothness_improvement:.2f}")
        
        if rmse_vs_reference is not None:
            print(f"  RMSE vs Reference: {rmse_vs_reference:.2f} dBm")
            print(f"  Correlation with Reference: {correlation_with_reference:.3f}")
    
    # Overall performance summary
    if rmse_scores:
        avg_rmse = np.mean(rmse_scores)
        print(f"\n=== Overall Performance ===")
        print(f"Average RMSE vs Reference: {avg_rmse:.2f} dBm")
        
        # Quality assessment
        if avg_rmse < 2.0:
            print("✅ EXCELLENT: Very close to reference filtering!")
        elif avg_rmse < 5.0:
            print("✅ GOOD: Close to reference filtering")
        elif avg_rmse < 10.0:
            print("⚠️  FAIR: Moderate similarity to reference")
        else:
            print("❌ POOR: Significant difference from reference")
    
    # Save results to file
    if save_results and evaluation_results:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        results_df = pd.DataFrame(evaluation_results)
        results_file = os.path.join(output_dir, "filter_performance_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    return evaluation_results

# Main execution
if __name__ == "__main__":
    # Set up file paths
    data_dir = "data"
    raw_file = os.path.join(data_dir, "raw_rssi.csv")
    filtered_file = os.path.join(data_dir, "filtered_rssi.csv")
    
    # Alternative if using .txt files:
    # raw_file = os.path.join(data_dir, "raw_rssi.txt")
    # filtered_file = os.path.join(data_dir, "filtered_rssi.txt")
    
    # Check if files exist
    if not os.path.exists(raw_file):
        print(f"Error: {raw_file} not found!")
        print(f"Please place your raw RSSI file in the {data_dir}/ directory")
        exit(1)
    
    if not os.path.exists(filtered_file):
        print(f"Warning: {filtered_file} not found!")
        print("Will run without reference comparison")
        filtered_file = None
    
    # Load data
    print("Loading RSSI data...")
    raw_data, filtered_data = load_rssi_data(raw_file, filtered_file)
    
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw data columns: {raw_data.columns.tolist()}")
    print(f"Unique devices in raw data: {raw_data['name'].nunique()}")
    
    if filtered_data is not None:
        print(f"Filtered data shape: {filtered_data.shape}")
        print(f"Unique devices in filtered data: {filtered_data['name'].nunique()}")
    
    # Apply Kalman filter to all devices
    print("\nApplying Kalman filter...")
    results = filter_all_devices_rssi(raw_data, 
                                    process_noise=1.0,    # Adjust these parameters
                                    measurement_noise=15.0) # based on your data
    
    # Evaluate performance
    evaluate_filter_performance(results, filtered_data)
    
    # Plot example comparison (first device)
    if results:
        first_device = list(results.keys())[0]
        print(f"\nPlotting comparison for: {first_device}")
        plot_comparison(results, first_device, filtered_data)
    
def save_filtered_data(results, output_dir="output", filename="kalman_filtered_results.csv"):
    """Save filtered results to CSV file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for saving
    save_data = []
    
    for device_key, data in results.items():
        device_name = data['device']
        rssi_type = data['rssi_type']
        
        for i, (timestamp, raw_val, filtered_val) in enumerate(zip(
            data['timestamps'], data['raw'], data['filtered'])):
            
            save_data.append({
                'device': device_name,
                'rssi_type': rssi_type,
                'timestamp': timestamp,
                'raw_rssi': raw_val,
                'kalman_filtered_rssi': filtered_val,
                'measurement_index': i
            })
    
    # Save to CSV
    df = pd.DataFrame(save_data)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Filtered data saved to: {filepath}")
    
    return filepath

def create_validation_report(results, reference_data=None, output_dir="output"):
    """Create a comprehensive validation report"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_file = os.path.join(output_dir, "validation_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("RSSI KALMAN FILTER VALIDATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write(f"Total devices processed: {len(results)}\n\n")
        
        # Summary statistics
        all_rmse = []
        all_noise_reduction = []
        
        for device_key, data in results.items():
            raw_rssi = np.array(data['raw'])
            filtered_rssi = np.array(data['filtered'])
            
            noise_reduction = np.std(raw_rssi) - np.std(filtered_rssi)
            all_noise_reduction.append(noise_reduction)
            
            # Calculate RMSE vs reference if available
            if reference_data is not None:
                device_name = data['device']
                rssi_col = data['rssi_type']
                ref_rssi, _ = process_device_rssi(reference_data, device_name, rssi_col)
                
                if ref_rssi is not None and len(ref_rssi) == len(filtered_rssi):
                    rmse = np.sqrt(np.mean((filtered_rssi - ref_rssi)**2))
                    all_rmse.append(rmse)
        
        if all_rmse:
            f.write("VALIDATION AGAINST REFERENCE DATA:\n")
            f.write(f"Average RMSE: {np.mean(all_rmse):.2f} dBm\n")
            f.write(f"Best RMSE: {np.min(all_rmse):.2f} dBm\n")
            f.write(f"Worst RMSE: {np.max(all_rmse):.2f} dBm\n")
            f.write(f"RMSE Standard Deviation: {np.std(all_rmse):.2f} dBm\n\n")
            
            # Quality assessment
            avg_rmse = np.mean(all_rmse)
            if avg_rmse < 2.0:
                f.write("QUALITY ASSESSMENT: EXCELLENT ✅\n")
                f.write("The Kalman filter is performing very well!\n\n")
            elif avg_rmse < 5.0:
                f.write("QUALITY ASSESSMENT: GOOD ✅\n")
                f.write("The Kalman filter is performing well.\n\n")
            elif avg_rmse < 10.0:
                f.write("QUALITY ASSESSMENT: FAIR ⚠️\n")
                f.write("The Kalman filter shows moderate performance. Consider tuning parameters.\n\n")
            else:
                f.write("QUALITY ASSESSMENT: POOR ❌\n")
                f.write("The Kalman filter needs parameter adjustment.\n\n")
        
        f.write("NOISE REDUCTION PERFORMANCE:\n")
        f.write(f"Average noise reduction: {np.mean(all_noise_reduction):.2f} dBm\n")
        f.write(f"Best noise reduction: {np.max(all_noise_reduction):.2f} dBm\n")
        f.write(f"Worst noise reduction: {np.min(all_noise_reduction):.2f} dBm\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        if all_rmse and np.mean(all_rmse) > 5.0:
            f.write("- Try reducing process_noise parameter for smoother output\n")
            f.write("- Try increasing measurement_noise parameter for more filtering\n")
        f.write("- Check individual device plots for specific issues\n")
        f.write("- Ensure timestamp ordering is correct\n")
    
    print(f"Validation report saved to: {report_file}")
    return report_file