import argparse
import sys
import os
import pandas as pd
import PyPDF2
from modules import normalizeText, predict

# Function to load data from PDF files
def load_file(filename):
    data = []
    path = os.path.join(os.getcwd())
    for filename in os.listdir(path):
        if filename.endswith('.pdf'):
            with open(os.path.join(path, filename), 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                filename = filename.split('.')[0]
                state = filename.split(' ')[0]
                city = filename[2:]
                raw_text = ''
                for page in range(2, len(pdf_reader.pages)):
                    raw_text += pdf_reader.pages[page].extract_text()
            data.append({'State': state, 'City': city, 'Raw Text': raw_text})
    df = pd.DataFrame(data)
    return df


# Main function
def main(args):
    df = load_file(args.document)

    df['Normalized Text'] = df['Raw Text'].apply(normalizeText.normalize_text)
    df = predict.cluster(df)

    df = predict.topic(df)

    if args.summarize == True:
        df['summary'] = df['Normalized Text'].apply(predict.summary)
    else:
        df['summary'] = "" 
    if args.keywords == True:
        df['keywords'] = df['Normalized Text'].apply(predict.keywords_text)
    else:
        df['keywords'] = ""

    #Printing the Output
    row = df.loc[0]

    city = row['City']
    state = row['State']
    clusterid = row['Cluster ID']
    if args.summarize == True:
        summary = row['summary']
    else:
        summary = ""
    if args.keywords == True:
        keywords = ', '.join(row['keywords'])
    else:
        keywords = ""

    output = f"[{city}, {state}] clusterid: {clusterid}\n[{city}, {state}] summary: {summary}\n[{city}, {state}] keywords: {keywords}"
    print(output)

    #Storing the .tsv file
    df = df.assign(city_state = df['City']+ ", " + df['State'])
    if os.path.isfile('smartcity_predict.tsv'):
        df.to_csv('smartcity_predict.tsv', sep='\t', index=False, mode='a', 
                header=False, 
                columns=['city_state','Raw Text', 'Normalized Text', 'Cluster ID', 'Topic_ids', 'summary', 'keywords'])
    else:
        df.to_csv('smartcity_predict.tsv', sep='\t', index=False, 
                columns=['city_state','Raw Text', 'Normalized Text', 'Cluster ID', 'Topic_ids', 'summary', 'keywords'])



# Function to get input from input terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart City Applicant Prediction')
    parser.add_argument('--document', type=str, required=True, help='Path to the new smart city applicant document')
    parser.add_argument('--summarize',action = 'store_true', help='Flag to indicate if document should be summarized')
    parser.add_argument('--keywords', action = 'store_true', 
                        help='Flag to indicate if keywords should be extracted from document')

    args = parser.parse_args()

    #To handle errors 
    try:
        sys.stderr.write("\n")
        with open(args.document) as f:
            pass
        main(args)

    except FileNotFoundError:
        print("Error: File not found")
        exit()