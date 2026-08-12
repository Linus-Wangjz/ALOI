import pandas as pd
import argparse

def filter_highest_throughput(args):
    # Load the dataset
    df = pd.read_csv(args.file_path)

    data_parallelism = df['Data parallelism'] if 'Data parallelism' in df else 1
    systolic_dim = df['Systolic dim'] if 'Systolic dim' in df else 1
    systolic_pim = df['Systolic pim'] if 'Systolic pim' in df else pd.Series(False, index=df.index)

    # Apply constraints. The fallbacks keep this compatible with both the
    # cent_dev CSV schema and CENT's single-data-parallel schema.
    df = df[
        (df['Device number'] * data_parallelism <= args.device)
        & (systolic_dim <= args.systolic_dim)
        & (df['Total power (W)'] <= args.power_limit)
    ]

    if args.seqlen > 0:
        df = df[(df['Sequence length'] == args.seqlen)]

    # df = df[(df['Split decoder across devices'] == False)]

    if args.systolic_pim:
        filtered_df = df[systolic_pim.loc[df.index] == True]
    else:
        filtered_df = df[systolic_pim.loc[df.index] == False]

    # Check if there are rows after filtering
    if filtered_df.empty:
        print(",,,,,,")
        return None

    # Find the row with the highest throughput
    # highest_throughput_row = filtered_df.loc[(filtered_df['Throughput (tokens/s)'] / filtered_df['Total power (W)']).idxmax()]
    highest_throughput_row = filtered_df.loc[filtered_df['Throughput (tokens/s)'].idxmax()]

    data_parallel = highest_throughput_row.get('Data parallelism', 1)
    split_decoder = highest_throughput_row.get('Split decoder across devices', False)
    row_systolic_dim = highest_throughput_row.get('Systolic dim', 1)
    print(f"{highest_throughput_row['Device number'] * data_parallel},{row_systolic_dim},,,{split_decoder},{highest_throughput_row['Throughput (tokens/s)']:.2f},{highest_throughput_row['Total power (W)']:.1f}")
    # print(f"{highest_throughput_row['Device number'] * highest_throughput_row['Data parallelism']},{highest_throughput_row['Systolic dim']},,,{highest_throughput_row['Split decoder across devices']},{highest_throughput_row['Throughput (tokens/s)']:.2f},{highest_throughput_row['Total power (W)']:.1f},{highest_throughput_row['Data parallelism']},{highest_throughput_row['Pipeline parallelism']}")

    return highest_throughput_row

def parse_args():
    parser = argparse.ArgumentParser(description='Filter CSV file for highest throughput configuration')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file to analyze')
    parser.add_argument('--systolic_pim', action='store_true', help='Use Systolic PIM configuration')
    parser.add_argument('--device', type=int, default=96, help='Maximum number of devices (default: 64)')
    parser.add_argument('--systolic_dim', type=int, default=4, help='Maximum systolic dimension (default: 8)')
    parser.add_argument('--power_limit', type=int, default=5600, help='Power limit in watts (default: 5000)')
    parser.add_argument('--seqlen', type=int, default=0, help='Sequence length (default: 0)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    filter_highest_throughput(args)
