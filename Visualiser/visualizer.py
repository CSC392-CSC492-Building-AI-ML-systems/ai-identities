import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from matplotlib.colors import ListedColormap
import json

n_max=30

def compare_models(data_files, top_n=n_max):
    all_data = pd.DataFrame()
    top_words_by_model = {}
    
    for file_path in data_files:
        df = pd.read_csv(file_path, header=None, names=['word', 'frequency'])
        df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce').fillna(0).astype(int)
    
        model_name = os.path.basename(file_path).replace('_cleaned.csv', '')
        df['model'] = model_name

        if df is not None:
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)

                model_name = df['model'].iloc[0]
                top_words_by_model[model_name] = df.sort_values('frequency', ascending=False).head(n_max)
    
    create_visualizations(all_data, top_words_by_model, top_n)

def create_visualizations(all_data, top_words_by_model, top_n=15):
    models = list(top_words_by_model.keys())
    num_models = len(models)
    
    n_rows = (num_models + 1) // 2  # not sure about this
    n_cols = 2
    
    plt.figure(figsize=(16, 6 * n_rows))
    
    for i, model in enumerate(models):
        plt.subplot(n_rows, n_cols, i + 1)
        
        df = top_words_by_model[model]
        sns.barplot(x='word', y='frequency', data=df)
        
        plt.title(f"Top {top_n} Words for {model}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.savefig('top_words_by_model.png')
    plt.close()
    
    all_top_words = set()
    for model in models:

        top_words = top_words_by_model[model]['word'].tolist()[:top_n]
        all_top_words.update(top_words)
    
    all_top_words_list = list(all_top_words)
    
    word_freq_matrix = pd.DataFrame(0, index=models, columns=all_top_words_list)
    
    for model in models:
        model_words = top_words_by_model[model]
        for _, row in model_words.iterrows():
            if row['word'] in all_top_words:
                word_freq_matrix.loc[model, row['word']] = row['frequency']
    
    normalized_matrix = word_freq_matrix.div(word_freq_matrix.sum(axis=1), axis=0)
    
    print("Normalized Frequencies for Each Word:")
    print(normalized_matrix)
    
    colors = ["purple"] * num_models # default, for the ass models
    cmap = ListedColormap(colors)
    
    highlight_matrix = pd.DataFrame(0, index=models, columns=all_top_words_list)
    for word in all_top_words_list:
        # index of highest/best guess model for that word
        max_model = normalized_matrix[word].idxmax()
        highlight_matrix.loc[max_model, word] = 1
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(normalized_matrix, annot=False, cmap=cmap, mask=highlight_matrix, vmin=0, vmax=1)
    
    sns.heatmap(normalized_matrix, annot=False, cmap=ListedColormap(["green"]), mask=~highlight_matrix.astype(bool), vmin=0, vmax=1, cbar=False)
    
    plt.title('Normalized Word Frequency Across Models (Top 15 Words)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('word_frequency_heatmap.png')
    plt.close()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(word_freq_matrix)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=models)
    pca_df.reset_index(inplace=True)
    pca_df.rename(columns={'index': 'model'}, inplace=True)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', s=100)
    
    for i, txt in enumerate(pca_df['model']):
        plt.annotate(txt, (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]), fontsize=12)
    
    plt.title('PCA of Model Word Distributions')
    plt.tight_layout()
    plt.savefig('model_pca.png')
    plt.close()

    heatmap_data = {
    'normalized_frequencies': normalized_matrix.to_dict(),
    'highest_frequency_model': {word: normalized_matrix[word].idxmax() for word in all_top_words_list}
    }

    with open('heatmap_data.json', 'w') as f:
        json.dump(heatmap_data, f, indent=2)

def clean_file(input_file, output_dir="cleaned_files"):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    last_non_empty_line = None
    for line in reversed(lines):
        if line.strip():
            last_non_empty_line = line
            break

    frequencies = []
    if last_non_empty_line:
        freq_values = [int(freq.strip()) for freq in last_non_empty_line.split(',') if freq.strip().isdigit()]
        frequencies.extend(freq_values)

    # got the regex from ai, aint no way I was doing this on my own
    adjectives = []
    for line in lines[:-1]:  # All lines except the last one
        if line.strip():  # Skip empty lines
            # Extract words using regex (match words separated by commas or periods)
            words = re.findall(r'[^,\s]+', line)  # Match any sequence of non-comma, non-whitespace characters
            adjectives.extend(words)

    if len(adjectives) != len(frequencies):
        print(f"Warning: Number of adjectives and frequencies do not match in {input_file}. Truncating to the shorter length.")
        min_length = min(len(adjectives), len(frequencies))
        adjectives = adjectives[:min_length]
        frequencies = frequencies[:min_length]

    data = pd.DataFrame({
        'word': adjectives,
        'frequency': frequencies
    })

    output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_cleaned.csv'))
    
    data.to_csv(output_file, index=False, header=False, encoding='utf-8')
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    # Add mistral and all those here too, for caution can add all
    # --> Example urls given, gotta edit these dont forget
    files_to_clean = [
        "C://Users//User//Downloads//phi-4_results.csv",
        "C://Users//User//Downloads//Mistral-Small-24B-Instruct-2501_results.csv",
        "C://Users//User//Downloads//gemma-3-27b-it_results.csv",
        "C://Users//User//Downloads//WizardLM-2-8x22B_results.csv"
    ]
    for file in files_to_clean:
        clean_file(file)

    cleaned_files = [
        "C://Users//User//Github_academic//CSC392//ai-identities//visualizer//cleaned_files//gpt-4o-mini_results_cleaned.csv",
        "C://Users//User//Github_academic//CSC392//ai-identities//visualizer//cleaned_files//Llama-3.3-70B-Instruct-Turbo_results_cleaned.csv",
        "C://Users//User//Github_academic//CSC392//ai-identities//visualizer//cleaned_files//phi-4_results_cleaned.csv",
        "C://Users//User//Github_academic//CSC392//ai-identities//visualizer//cleaned_files//Mistral-Small-24B-Instruct-2501_results_cleaned.csv",
        "C://Users//User//Github_academic//CSC392//ai-identities//visualizer//cleaned_files//gemma-3-27b-it_results_cleaned.csv",
        "C://Users//User//Github_academic//CSC392//ai-identities//visualizer//cleaned_files//WizardLM-2-8x22B_results_cleaned.csv"
    ]
    compare_models(cleaned_files)