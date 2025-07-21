import pandas as pd
from sklearn.model_selection import train_test_split
from clean_text import clean_text  # Import clean_text function

def split_data(df, label_col='toxic'):
    # Apply text cleaning to 'text' column
    df['text'] = df['text'].fillna('')

    # split into train(80%) and temp(20%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    # split temp into validation(10%) and test(10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[label_col])
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    # Load your raw dataset file here
    df = pd.read_csv('data/merged_dataset_final.csv')

    train, val, test = split_data(df)
    print(f'Train size: {len(train)}')
    print(f'Validation size: {len(val)}')
    print(f'Test size: {len(test)}')

    # Save the cleaned splits to CSV files
    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)
