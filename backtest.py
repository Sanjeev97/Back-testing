import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Start timing the execution
start_time = time.time()

# Load and preprocess the market data
def preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Sort by timestamp
    df = df.sort_values('ts_event')
    
    # Keep only the first message per publisher_id per unique ts_event
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'])
    
    # Extract relevant columns
    df = df[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']]
    
    # Remove rows with missing values in key columns
    df = df.dropna(subset=['ask_px_00', 'ask_sz_00'])
    
    # Convert timestamps to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Only keep rows with valid ask price and size
    df = df[(df['ask_px_00'] > 0) & (df['ask_sz_00'] > 0)]
    
    return df

# Implement the allocator as per the pseudocode
def allocate(order_size, venues, lambda_over, lambda_under, theta_queue):
    step = 100  # search in 100-share chunks
    splits = [[]]  # start with an empty allocation list
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v]['ask_size'])
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits
    
    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    
    return best_split, best_cost

def compute_cost(split, venues, order_size, lambda_o, lambda_u, theta):
    executed = 0
    cash_spent = 0
    
    for i in range(len(venues)):
        exe = min(split[i], venues[i]['ask_size'])
        executed += exe
        cash_spent += exe * (venues[i]['ask'] + venues[i]['fee'])
        maker_rebate = max(split[i] - exe, 0) * venues[i]['rebate']
        cash_spent -= maker_rebate
    
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta * (underfill + overfill)
    cost_pen = lambda_u * underfill + lambda_o * overfill
    
    return cash_spent + risk_pen + cost_pen

# Back-testing function
def backtest(data, params, order_size=5000):
    lambda_over, lambda_under, theta_queue = params
    
    # Group data by timestamp
    grouped = data.groupby('ts_event')
    
    total_shares_executed = 0
    total_cash_spent = 0
    remaining_order = order_size
    execution_details = []
    
    # Track execution state
    execution_state = []
    
    # Process each timestamp
    for ts, group in grouped:
        if remaining_order <= 0:
            break
            
        # Create venues for the allocator
        venues = []
        for _, row in group.iterrows():
            venues.append({
                'ask': row['ask_px_00'],
                'ask_size': row['ask_sz_00'],
                'fee': 0.0,  # Assuming no fees for simplicity
                'rebate': 0.0,  # Assuming no rebates for simplicity
                'publisher_id': row['publisher_id']
            })
        
        # Skip if no venues are available
        if not venues:
            continue
            
        # Use allocator to determine order splits
        split, cost = allocate(remaining_order, venues, lambda_over, lambda_under, theta_queue)
        
        # Execute the orders according to the split
        shares_executed_at_ts = 0
        cash_spent_at_ts = 0
        
        for i, alloc in enumerate(split):
            exe = min(alloc, venues[i]['ask_size'])
            shares_executed_at_ts += exe
            cash_spent_at_ts += exe * venues[i]['ask']
            
            # Record execution details
            if exe > 0:
                execution_details.append({
                    'timestamp': ts,
                    'venue': venues[i]['publisher_id'],
                    'price': venues[i]['ask'],
                    'shares': exe,
                    'cost': exe * venues[i]['ask']
                })
        
        # Update totals
        total_shares_executed += shares_executed_at_ts
        total_cash_spent += cash_spent_at_ts
        remaining_order -= shares_executed_at_ts
        
        # Record state for tracking
        execution_state.append({
            'timestamp': ts,
            'shares_executed': shares_executed_at_ts,
            'cash_spent': cash_spent_at_ts,
            'total_shares': total_shares_executed,
            'total_cost': total_cash_spent,
            'remaining_order': remaining_order
        })
    
    # Calculate average price if any shares were executed
    avg_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    result = {
        'shares_executed': total_shares_executed,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'execution_details': execution_details,
        'execution_state': execution_state
    }
    
    return result

# Implement baseline strategies
def best_ask_strategy(data, order_size=5000):
    # Group data by timestamp
    grouped = data.groupby('ts_event')
    
    total_shares_executed = 0
    total_cash_spent = 0
    remaining_order = order_size
    execution_state = []
    
    # Process each timestamp
    for ts, group in grouped:
        if remaining_order <= 0:
            break
            
        # Find the best ask price
        best_ask_row = group.loc[group['ask_px_00'].idxmin()]
        
        # Execute at the best ask
        exe = min(remaining_order, best_ask_row['ask_sz_00'])
        cash_spent = exe * best_ask_row['ask_px_00']
        
        # Update totals
        total_shares_executed += exe
        total_cash_spent += cash_spent
        remaining_order -= exe
        
        # Record state
        execution_state.append({
            'timestamp': ts,
            'shares_executed': exe,
            'cash_spent': cash_spent,
            'total_shares': total_shares_executed,
            'total_cost': total_cash_spent,
            'remaining_order': remaining_order
        })
    
    # Calculate average price
    avg_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'shares_executed': total_shares_executed,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'execution_state': execution_state
    }

def twap_strategy(data, order_size=5000, bucket_seconds=60):
    # Group data by timestamp
    data['bucket'] = data['ts_event'].dt.floor(f'{bucket_seconds}S')
    grouped = data.groupby('bucket')
    
    # Calculate shares per bucket
    n_buckets = len(grouped)
    shares_per_bucket = order_size / n_buckets
    
    total_shares_executed = 0
    total_cash_spent = 0
    execution_state = []
    
    # Process each bucket
    for bucket, group in grouped:
        # Group by timestamp within the bucket
        ts_groups = group.groupby('ts_event')
        bucket_shares_remaining = min(shares_per_bucket, order_size - total_shares_executed)
        
        if bucket_shares_remaining <= 0:
            break
            
        # Process each timestamp in the bucket
        for ts, ts_group in ts_groups:
            if bucket_shares_remaining <= 0:
                break
                
            # Sort venues by ask price
            sorted_venues = ts_group.sort_values('ask_px_00')
            
            # Execute orders
            for _, venue in sorted_venues.iterrows():
                exe = min(bucket_shares_remaining, venue['ask_sz_00'])
                cash_spent = exe * venue['ask_px_00']
                
                # Update totals
                total_shares_executed += exe
                total_cash_spent += cash_spent
                bucket_shares_remaining -= exe
                
                # Record state
                execution_state.append({
                    'timestamp': ts,
                    'shares_executed': exe,
                    'cash_spent': cash_spent,
                    'total_shares': total_shares_executed,
                    'total_cost': total_cash_spent,
                    'remaining_order': order_size - total_shares_executed
                })
                
                if bucket_shares_remaining <= 0:
                    break
    
    # Calculate average price
    avg_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'shares_executed': total_shares_executed,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'execution_state': execution_state
    }

def vwap_strategy(data, order_size=5000):
    # Group data by timestamp
    grouped = data.groupby('ts_event')
    
    total_shares_executed = 0
    total_cash_spent = 0
    remaining_order = order_size
    execution_state = []
    
    # Calculate total volume first
    total_volume = data['ask_sz_00'].sum()
    
    # Process each timestamp
    for ts, group in grouped:
        if remaining_order <= 0:
            break
            
        # Calculate volume weight for this timestamp
        ts_volume = group['ask_sz_00'].sum()
        weight = ts_volume / total_volume
        
        # Calculate shares to execute at this timestamp
        shares_to_execute = min(int(order_size * weight), remaining_order)
        
        # Sort venues by ask price
        sorted_venues = group.sort_values('ask_px_00')
        
        # Execute orders
        shares_executed_at_ts = 0
        cash_spent_at_ts = 0
        
        for _, venue in sorted_venues.iterrows():
            exe = min(shares_to_execute - shares_executed_at_ts, venue['ask_sz_00'])
            cash_spent = exe * venue['ask_px_00']
            
            shares_executed_at_ts += exe
            cash_spent_at_ts += cash_spent
            
            if shares_executed_at_ts >= shares_to_execute:
                break
        
        # Update totals
        total_shares_executed += shares_executed_at_ts
        total_cash_spent += cash_spent_at_ts
        remaining_order -= shares_executed_at_ts
        
        # Record state
        execution_state.append({
            'timestamp': ts,
            'shares_executed': shares_executed_at_ts,
            'cash_spent': cash_spent_at_ts,
            'total_shares': total_shares_executed,
            'total_cost': total_cash_spent,
            'remaining_order': remaining_order
        })
    
    # Calculate average price
    avg_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'shares_executed': total_shares_executed,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'execution_state': execution_state
    }

# Parameter search function
def search_optimal_parameters(data, order_size=5000):
    # Define parameter search grid
    lambda_over_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    lambda_under_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    theta_queue_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    best_result = None
    best_params = None
    best_avg_price = float('inf')
    
    # Search across parameter grid
    for lambda_over in lambda_over_values:
        for lambda_under in lambda_under_values:
            for theta_queue in theta_queue_values:
                params = (lambda_over, lambda_under, theta_queue)
                
                # Run backtest with these parameters
                result = backtest(data, params, order_size)
                
                # Check if this is the best result so far
                if result['shares_executed'] > 0 and result['avg_price'] < best_avg_price:
                    best_result = result
                    best_params = params
                    best_avg_price = result['avg_price']
    
    return best_params, best_result

# Plot function
def create_cumulative_cost_plot(smart_router_state, best_ask_state, twap_state, vwap_state):
    plt.figure(figsize=(10, 6))
    
    # Extract data for plotting
    smart_router_times = [state['timestamp'] for state in smart_router_state]
    smart_router_costs = [state['total_cost'] for state in smart_router_state]
    
    best_ask_times = [state['timestamp'] for state in best_ask_state]
    best_ask_costs = [state['total_cost'] for state in best_ask_state]
    
    twap_times = [state['timestamp'] for state in twap_state]
    twap_costs = [state['total_cost'] for state in twap_state]
    
    vwap_times = [state['timestamp'] for state in vwap_state]
    vwap_costs = [state['total_cost'] for state in vwap_state]
    
    # Plot each strategy
    plt.plot(smart_router_times, smart_router_costs, label='Smart Router')
    plt.plot(best_ask_times, best_ask_costs, label='Best Ask')
    plt.plot(twap_times, twap_costs, label='TWAP')
    plt.plot(vwap_times, vwap_costs, label='VWAP')
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative Cost ($)')
    plt.title('Cumulative Execution Cost Comparison')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis as time
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.savefig('results.png')

# Main execution
if __name__ == "__main__":
    # Load and preprocess the data
    data = preprocess_data('l1_day.csv')
    
    # Define order size
    order_size = 5000
    
    # Find optimal parameters
    best_params, best_result = search_optimal_parameters(data, order_size)
    
    # Run baseline strategies
    best_ask_result = best_ask_strategy(data, order_size)
    twap_result = twap_strategy(data, order_size)
    vwap_result = vwap_strategy(data, order_size)
    
    # Calculate basis point savings
    best_ask_bps = ((best_ask_result['avg_price'] / best_result['avg_price']) - 1) * 10000
    twap_bps = ((twap_result['avg_price'] / best_result['avg_price']) - 1) * 10000
    vwap_bps = ((vwap_result['avg_price'] / best_result['avg_price']) - 1) * 10000
    
    # Create result JSON
    result_json = {
        "best_parameters": {
            "lambda_over": best_params[0],
            "lambda_under": best_params[1],
            "theta_queue": best_params[2]
        },
        "smart_router": {
            "cash_spent": best_result['cash_spent'],
            "avg_price": best_result['avg_price']
        },
        "best_ask": {
            "cash_spent": best_ask_result['cash_spent'],
            "avg_price": best_ask_result['avg_price']
        },
        "twap": {
            "cash_spent": twap_result['cash_spent'],
            "avg_price": twap_result['avg_price']
        },
        "vwap": {
            "cash_spent": vwap_result['cash_spent'],
            "avg_price": vwap_result['avg_price']
        },
        "savings_bps": {
            "vs_best_ask": best_ask_bps,
            "vs_twap": twap_bps,
            "vs_vwap": vwap_bps
        },
        "execution_time_seconds": time.time() - start_time
    }
    
    # Create and save plot
    create_cumulative_cost_plot(
        best_result['execution_state'],
        best_ask_result['execution_state'],
        twap_result['execution_state'],
        vwap_result['execution_state']
    )
    
    # Print JSON result
    print(json.dumps(result_json, indent=2))