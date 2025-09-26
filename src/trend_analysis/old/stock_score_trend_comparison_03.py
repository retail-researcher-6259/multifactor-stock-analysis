"""
Enhanced Script to find tickers that appear in top N of both ranking files
With Market Regime Support for Steady_Growth and Crisis_Bear analysis
- Reads top_ranked_stocks file (sorted by Score)
- Reads stability_analysis_results file (sorted by stability_adjusted_score)
- Returns tickers that are in top N of both files
- ENHANCED: Includes Sector and Industry information from the ranked stocks file
- V03: Added market regime selector for different market conditions
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

def find_common_top150_tickers_enhanced(ranked_file, stability_file, top_n=150):
    """
    Find tickers that appear in top N positions of both files
    Enhanced version that includes Sector and Industry information

    Parameters:
    -----------
    ranked_file : str or Path
        Path to top_ranked_stocks CSV file
    stability_file : str or Path
        Path to stability_analysis_results CSV file
    top_n : int
        Number of top entries to consider (default: 150)

    Returns:
    --------
    pd.DataFrame
        DataFrame with common tickers, their rankings in both files, and sector/industry info
    """

    # Read the ranked stocks file
    print(f"Reading ranked stocks from: {ranked_file}")
    ranked_df = pd.read_csv(ranked_file)

    # Read the stability analysis file
    print(f"Reading stability analysis from: {stability_file}")
    stability_df = pd.read_csv(stability_file)

    # Get top N from ranked stocks (sorted by Score)
    # The file should already be sorted, but let's ensure it
    ranked_topN = ranked_df.nlargest(top_n, 'Score')
    ranked_topN['rank_in_ranked'] = range(1, len(ranked_topN) + 1)

    # Get top N from stability analysis (sorted by stability_adjusted_score)
    # The file should already be sorted, but let's ensure it
    stability_topN = stability_df.nlargest(top_n, 'stability_adjusted_score')
    stability_topN['rank_in_stability'] = range(1, len(stability_topN) + 1)

    # Find common tickers
    ranked_tickers = set(ranked_topN['Ticker'])
    stability_tickers = set(stability_topN['ticker'])
    common_tickers = ranked_tickers.intersection(stability_tickers)

    print(f"\nFound {len(common_tickers)} tickers in top {top_n} of both files")

    # Create a lookup dictionary for sector and industry information
    # This will help us get sector/industry for any ticker in the ranked file
    sector_industry_lookup = {}
    for _, row in ranked_df.iterrows():
        sector_industry_lookup[row['Ticker']] = {
            'Sector': row['Sector'],
            'Industry': row['Industry'],
            'CompanyName': row['CompanyName'],
            'Country': row['Country']
        }

    # Create result dataframe with details from both files plus sector/industry
    results = []

    for ticker in common_tickers:
        # Get info from ranked file
        ranked_row = ranked_topN[ranked_topN['Ticker'] == ticker].iloc[0]
        ranked_position = ranked_row['rank_in_ranked']
        ranked_score = ranked_row['Score']

        # Get info from stability file
        stability_row = stability_topN[stability_topN['ticker'] == ticker].iloc[0]
        stability_position = stability_row['rank_in_stability']
        stability_score = stability_row['stability_adjusted_score']
        avg_score = stability_row['avg_score']
        linear_r2 = stability_row['linear_r2']
        linear_slope = stability_row['linear_slope']
        recommendation = stability_row['recommendation']

        # Get sector and industry information
        ticker_info = sector_industry_lookup.get(ticker, {
            'Sector': 'Unknown',
            'Industry': 'Unknown',
            'CompanyName': 'Unknown',
            'Country': 'Unknown'
        })

        results.append({
            'Ticker': ticker,
            'CompanyName': ticker_info['CompanyName'],
            'Country': ticker_info['Country'],
            'Rank_in_Ranked': ranked_position,
            'Score': ranked_score,
            'Rank_in_Stability': stability_position,
            'Stability_Adjusted_Score': stability_score,
            'Avg_Score': avg_score,
            'Linear_R2': linear_r2,
            'Linear_Slope': linear_slope,
            'Recommendation': recommendation,
            'Combined_Rank': ranked_position + stability_position,  # Lower is better
            'Sector': ticker_info['Sector'],
            'Industry': ticker_info['Industry']
        })

    # Convert to DataFrame and sort by combined rank
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Combined_Rank')

    return results_df

def analyze_sector_distribution(results_df):
    """
    Analyze the sector and industry distribution of common top performers

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from find_common_top150_tickers_enhanced

    Returns:
    --------
    dict
        Dictionary containing sector and industry analysis
    """
    print("\n" + "="*80)
    print("SECTOR AND INDUSTRY ANALYSIS")
    print("="*80)

    # Sector distribution
    sector_counts = results_df['Sector'].value_counts()
    print(f"\nSector Distribution (Top {len(results_df)} Common Performers):")
    print("-"*50)
    for sector, count in sector_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"{sector:<30} {count:>3} ({percentage:>5.1f}%)")

    # Industry distribution (top 10)
    industry_counts = results_df['Industry'].value_counts()
    print(f"\nTop 10 Industries:")
    print("-"*50)
    for industry, count in industry_counts.head(10).items():
        percentage = (count / len(results_df)) * 100
        print(f"{industry:<40} {count:>3} ({percentage:>5.1f}%)")

    # Country distribution
    country_counts = results_df['Country'].value_counts()
    print(f"\nCountry Distribution:")
    print("-"*50)
    for country, count in country_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"{country:<30} {count:>3} ({percentage:>5.1f}%)")

    return {
        'sector_distribution': sector_counts,
        'industry_distribution': industry_counts,
        'country_distribution': country_counts
    }

def show_sector_top_performers(results_df, top_n_per_sector=3):
    """
    Show top performers by sector

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from find_common_top150_tickers_enhanced
    top_n_per_sector : int
        Number of top performers to show per sector
    """
    print(f"\n" + "="*80)
    print(f"TOP {top_n_per_sector} PERFORMERS BY SECTOR")
    print("="*80)

    for sector in results_df['Sector'].unique():
        sector_df = results_df[results_df['Sector'] == sector].head(top_n_per_sector)

        print(f"\n🏢 {sector.upper()} ({len(results_df[results_df['Sector'] == sector])} total)")
        print("-"*60)

        for _, row in sector_df.iterrows():
            print(f"{row['Ticker']:<6} {row['CompanyName'][:40]:<42} "
                  f"Rank: {row['Combined_Rank']:>3} "
                  f"Score: {row['Score']:>6.1f}")

def main(ranked_file, stability_file, top_n=150, output_dir="./results"):
    """
    Main function to execute the enhanced analysis

    Parameters:
    -----------
    ranked_file : str
        Path to top_ranked_stocks CSV file
    stability_file : str
        Path to stability_analysis_results CSV file
    top_n : int
        Number of top entries to consider
    output_dir : str
        Directory to save output files
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Find common top N tickers with enhanced information
        common_df = find_common_top150_tickers_enhanced(ranked_file, stability_file, top_n=top_n)

        # Display results
        print("\n" + "="*80)
        print(f"TICKERS IN TOP {top_n} OF BOTH RANKINGS (WITH SECTOR/INDUSTRY)")
        print("="*80)
        print(f"\nTotal tickers found: {len(common_df)}")
        print(f"\nTop 20 by combined rank (from top {top_n} of each list):")
        print("-"*80)

        # Display top 20 with sector/industry
        display_cols = ['Ticker', 'CompanyName', 'Sector', 'Industry',
                        'Rank_in_Ranked', 'Score', 'Rank_in_Stability',
                        'Stability_Adjusted_Score', 'Combined_Rank']

        # Truncate company name for better display
        display_df = common_df.copy()
        display_df['CompanyName'] = display_df['CompanyName'].str[:30]
        display_df['Industry'] = display_df['Industry'].str[:25]

        print(display_df[display_cols].head(20).to_string(index=False))

        # Save full results with all new columns
        today = datetime.now().strftime("%m%d")  # e.g. 0808
        output_file = f"{output_dir}/common_top{top_n}_tickers_{today}.csv"
        common_df.to_csv(output_file, index=False)
        print(f"\n✅ Full results saved to: {output_file}")

        # Perform sector and industry analysis
        analysis_results = analyze_sector_distribution(common_df)

        # Show top performers by sector
        show_sector_top_performers(common_df, top_n_per_sector=3)

        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Average rank in ranked list: {common_df['Rank_in_Ranked'].mean():.1f}")
        print(f"Average rank in stability list: {common_df['Rank_in_Stability'].mean():.1f}")
        print(f"Best combined rank: {common_df['Combined_Rank'].min()}")
        print(f"Worst combined rank: {common_df['Combined_Rank'].max()}")

        # Show tickers that are top 50 in both lists
        top50_both = common_df[(common_df['Rank_in_Ranked'] <= 50) &
                               (common_df['Rank_in_Stability'] <= 50)]
        if len(top50_both) > 0:
            print(f"\n🌟 Tickers in TOP 50 of BOTH lists ({len(top50_both)} found):")
            print("-"*80)
            top50_display = top50_both.copy()
            top50_display['CompanyName'] = top50_display['CompanyName'].str[:25]
            print(top50_display[['Ticker', 'CompanyName', 'Sector', 'Rank_in_Ranked',
                                 'Rank_in_Stability', 'Score', 'Stability_Adjusted_Score',
                                 'Recommendation']].to_string(index=False))

        # Show best performers by recommendation
        print(f"\n📈 BREAKDOWN BY RECOMMENDATION:")
        print("-"*50)
        recommendation_counts = common_df['Recommendation'].value_counts()
        for rec, count in recommendation_counts.items():
            percentage = (count / len(common_df)) * 100
            print(f"{rec:<35} {count:>3} ({percentage:>5.1f}%)")

        return common_df, analysis_results

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
        print("Please ensure both CSV files are in the correct location")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check that the CSV files have the expected column names")
        return None, None

if __name__ == "__main__":
    # ===== CONFIGURATION SECTION =====
    # Modify these variables for easy customization

    # MARKET REGIME SELECTOR
    # 0 = Steady_Growth, 1 = Crisis_Bear
    MARKET_REGIME = 0
    FILE_DATE = "0822"

    # Define regime-specific settings
    if MARKET_REGIME == 0:
        REGIME_NAME = "Steady_Growth"
        # FILE_DATE = "0809"  # Adjust this date as needed
    elif MARKET_REGIME == 1:
        REGIME_NAME = "Crisis_Bear"
        # FILE_DATE = "0809"  # Adjust this date as needed
    else:
        raise ValueError("MARKET_REGIME must be 0 (Steady_Growth) or 1 (Crisis_Bear)")

    # File paths (regime-specific)
    RANKED_FILE = f"./ranked_lists/{REGIME_NAME}/top_ranked_stocks_{FILE_DATE}.csv"
    STABILITY_FILE = f"./results/{REGIME_NAME}/stability_analysis_results_{FILE_DATE}.csv"

    # Output directory (regime-specific)
    OUTPUT_DIRECTORY = f"./results/{REGIME_NAME}"

    # Number of top entries to consider from each file
    TOP_N = 150

    print(f"\n{'='*60}")
    print(f"STOCK COMPARISON - MARKET REGIME: {REGIME_NAME}")
    print(f"{'='*60}")
    print(f"File Date: {FILE_DATE}")
    print(f"Ranked File: {RANKED_FILE}")
    print(f"Stability File: {STABILITY_FILE}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print(f"Top N: {TOP_N}")
    print(f"{'='*60}\n")

    # ===== END CONFIGURATION =====

    # Run the enhanced analysis with the configured parameters
    results_df, analysis_results = main(
        RANKED_FILE,
        STABILITY_FILE,
        TOP_N,
        OUTPUT_DIRECTORY
    )

    if results_df is not None:
        print(f"\n✅ Analysis complete for {REGIME_NAME} regime!")
        print(f"📁 Check the output file: {OUTPUT_DIRECTORY}/common_top{TOP_N}_tickers_*.csv")
        print(f"{'='*60}")