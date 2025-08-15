import pandas as pd
from data_loader import load_raw_data
from data_splitter import bin_temperatures
from llm_meta_data import load_llm_meta_data
import yaml
import re
from tabulate import tabulate


def detect_refusal(response: str) -> bool:
    """
    Heuristic to detect if a response is a refusal.
    Criteria: Short length, or contains refusal keywords.
    """
    refusal_keywords = ['sorry', 'cannot', 'refuse', 'unable', 'not allowed',
                        'instructed not to']
    if len(response) < 1000:
        return True
    if any(re.search(keyword, response.lower()) for keyword in refusal_keywords):
        return True
    return False


def analyze_refusals(data: dict[str, pd.DataFrame], temp_bins: dict,
                     meta_path: str = '../configs/llm_set.json') -> dict:
    """
    Analyze refusal patterns across the dataset.
    Returns a dict with statistics.
    """
    all_df = pd.concat([df.assign(model=model) for model, df in data.items()])
    all_df = bin_temperatures(all_df, temp_bins)  # Add temp_bin column
    all_df['model'] = all_df['model'].str.replace('_', '/', n=1)

    # Detect refusals
    all_df['is_refusal'] = all_df['response'].apply(detect_refusal)

    # Add LLM family using metadata
    meta_map = load_llm_meta_data(meta_path)
    all_df['family'] = all_df['model'].apply(
        lambda m: meta_map.get(m, {}).get('family', 'unknown'))

    stats = {}

    # Overall refusal rate
    total_responses = len(all_df)
    refusal_count = all_df['is_refusal'].sum()
    stats['overall'] = {
        'total_responses': total_responses,
        'refusal_count': refusal_count,
        'refusal_rate': refusal_count / total_responses if total_responses > 0 else 0
    }

    # Per system prompt
    if 'system_prompt' in all_df.columns:
        per_system_prompt = all_df.groupby('system_prompt')['is_refusal'].agg(
            ['count', 'sum', 'mean']).rename(
            columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
        stats['per_system_prompt'] = per_system_prompt.sort_values('refusal_rate',
                                                                   ascending=False).reset_index()

    # Per LLM
    per_llm = all_df.groupby('model')['is_refusal'].agg(['count', 'sum', 'mean']).rename(
        columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
    stats['per_llm'] = per_llm.sort_values('refusal_rate', ascending=False).reset_index()

    # Per temp bin
    per_temp_bin = all_df.groupby('temp_bin')['is_refusal'].agg(
        ['count', 'sum', 'mean']).rename(
        columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
    stats['per_temp_bin'] = per_temp_bin.sort_values('refusal_rate',
                                                     ascending=False).reset_index()

    # Per (LLM, system prompt)
    if 'system_prompt' in all_df.columns:
        per_llm_sys = all_df.groupby(['model', 'system_prompt'])['is_refusal'].agg(
            ['count', 'sum', 'mean']).rename(
            columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
        stats['per_llm_system_prompt'] = per_llm_sys.sort_values('refusal_rate',
                                                                 ascending=False).reset_index()

    # Per (system prompt, temp bin)
    if 'system_prompt' in all_df.columns:
        per_sys_temp = all_df.groupby(['system_prompt', 'temp_bin'])['is_refusal'].agg(
            ['count', 'sum', 'mean']).rename(
            columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
        stats['per_system_prompt_temp_bin'] = per_sys_temp.sort_values('refusal_rate',
                                                                       ascending=False).reset_index()

    # Per (LLM, temp bin)
    per_llm_temp = all_df.groupby(['model', 'temp_bin'])['is_refusal'].agg(
        ['count', 'sum', 'mean']).rename(
        columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
    stats['per_llm_temp_bin'] = per_llm_temp.sort_values('refusal_rate',
                                                         ascending=False).reset_index()

    # Per LLM family
    per_family = all_df.groupby('family')['is_refusal'].agg(
        ['count', 'sum', 'mean']).rename(
        columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
    stats['per_family'] = per_family.sort_values('refusal_rate',
                                                 ascending=False).reset_index()

    # Per (LLM family, system prompt)
    if 'system_prompt' in all_df.columns:
        per_family_sys = all_df.groupby(['family', 'system_prompt'])['is_refusal'].agg(
            ['count', 'sum', 'mean']).rename(
            columns={'sum': 'refusal_count', 'mean': 'refusal_rate'})
        stats['per_family_system_prompt'] = per_family_sys.sort_values('refusal_rate',
                                                                       ascending=False).reset_index()

    # Top 10 highest refusal rates for (LLM, system prompt) pairs
    if 'per_llm_system_prompt' in stats:
        stats['top_20_llm_system_prompt'] = stats['per_llm_system_prompt'].head(20)

    # Cross-tab: Average refusal rate by (system prompt, LLM)
    if 'system_prompt' in all_df.columns:
        cross_tab = pd.pivot_table(all_df, values='is_refusal', index='system_prompt',
                                   columns='model', aggfunc='mean').fillna(0)
        stats['cross_tab'] = cross_tab.sort_index().reset_index()  # For table formatting

    return stats


def save_refusal_report(stats: dict, output_path: str):
    """
    Save detailed refusal analysis report to a text file with nicely formatted tables using tabulate.
    """
    with open(output_path, 'w') as f:
        f.write("Final CLF Dataset Refusal Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        # Summary Insights
        f.write("Key Insights:\n")
        overall = stats['overall']
        f.write(f"- Overall Refusal Rate: {overall['refusal_rate']:.2%} "
                f"({overall['refusal_count']} out of {overall['total_responses']})\n")
        if 'per_system_prompt' in stats:
            top_sys = stats['per_system_prompt'].iloc[0]['system_prompt']
            f.write(f"- Highest Refusing System Prompt: {top_sys[:50]}... "
                    f"(Rate: {stats['per_system_prompt'].iloc[0]['refusal_rate']:.2%})\n")
        top_llm = stats['per_llm'].iloc[0]['model']
        f.write(f"- Highest Refusing LLM: {top_llm} "
                f"(Rate: {stats['per_llm'].iloc[0]['refusal_rate']:.2%})\n")
        # Highest refusing family
        if 'per_family' in stats:
            top_family = stats['per_family'].iloc[0]['family']
            f.write(f"- Highest Refusing Family: {top_family} "
                    f"(Rate: {stats['per_family'].iloc[0]['refusal_rate']:.2%})\n")
        f.write("\n")

        # Overall Table
        f.write("Overall Statistics:\n")
        table_data = [['Total Responses', overall['total_responses']],
                      ['Refusal Count', overall['refusal_count']],
                      ['Refusal Rate', f"{overall['refusal_rate']:.2%}"]]
        f.write(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='simple') + "\n\n")

        # Per system prompt table
        if 'per_system_prompt' in stats:
            f.write("Refusal Rates per System Prompt (Truncated to 100 chars):\n")
            df = stats['per_system_prompt'].copy()
            df['system_prompt'] = df['system_prompt'].apply(
                lambda x: x[:100] + '...' if len(x) > 100 else x)
            df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
            f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per LLM table
        f.write("Refusal Rates per LLM:\n")
        df = stats['per_llm'].copy()
        df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
        f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per temp bin table
        f.write("Refusal Rates per Temp Bin:\n")
        df = stats['per_temp_bin'].copy()
        df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
        f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per LLM family table
        if 'per_family' in stats:
            f.write("Refusal Rates per LLM Family:\n")
            df = stats['per_family'].copy()
            df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
            f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per (LLM, system prompt) table (top 10)
        if 'top_20_llm_system_prompt' in stats:
            f.write("Top 20 Refusal Rates per (LLM, System Prompt) (Prompt Trunc to 100 chars):\n")
            df = stats['top_20_llm_system_prompt'].copy()
            df['system_prompt'] = df['system_prompt'].apply(
                lambda x: x[:100] + '...' if len(x) > 100 else x)
            df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
            f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per (system prompt, temp bin) table
        if 'per_system_prompt_temp_bin' in stats:
            f.write("Refusal Rates per (System Prompt, Temp Bin) (Prompt Trunc to 100 chars):\n")
            df = stats['per_system_prompt_temp_bin'].copy()
            df['system_prompt'] = df['system_prompt'].apply(
                lambda x: x[:100] + '...' if len(x) > 100 else x)
            df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
            f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per (LLM, temp bin) table
        f.write("Refusal Rates per (LLM, Temp Bin):\n")
        df = stats['per_llm_temp_bin'].copy()
        df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
        f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Per (LLM family, system Prompt) full table (if not too large; otherwise, limit)
        if 'per_family_system_prompt' in stats:
            f.write("Refusal Rates per (LLM Family, System Prompt) (Prompt Trunc to 100 chars):\n")
            df = stats['per_family_system_prompt'].copy()
            df['system_prompt'] = df['system_prompt'].apply(
                lambda x: x[:100] + '...' if len(x) > 100 else x)
            df['refusal_rate'] = df['refusal_rate'].apply(lambda x: f"{x:.2%}")
            f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

        # Cross-Tab Table (Average Refusal Rate by System Prompt and LLM, Top 5 System Prompts)
        if 'cross_tab' in stats:
            f.write("Cross-Tab: Avg Refusal Rate by System Prompt and LLM (Top 5 System Prompts, Rates as %):\n")
            df = stats['cross_tab'].head(5).copy()  # Limit to top 5 for brevity
            df['system_prompt'] = df['system_prompt'].apply(
                lambda x: x[:100] + '...' if len(x) > 100 else x)
            for col in df.columns[1:]:
                df[col] = df[col].apply(lambda x: f"{x:.2%}")
            f.write(tabulate(df, headers='keys', tablefmt='simple', showindex=False) + "\n\n")

    print(f"Refusal analysis report saved to {output_path}")


if __name__ == '__main__':
    config = yaml.safe_load(open('../configs/data_config.yaml'))
    raw_path = '../data/final_clf_dataset_raw_data/'
    data, _, _, _ = load_raw_data(raw_path)
    stats = analyze_refusals(data, config['final_clf_dataset_temp_bins'])
    save_refusal_report(stats, '../results/final_clf_dataset_analysis/final_clf_refusal_analysis.txt')