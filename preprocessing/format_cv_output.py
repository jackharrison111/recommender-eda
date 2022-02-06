import json 
import pandas as pd 
import numpy as np
import gensim.parsing.preprocessing as pp

# # Download Wordnet through NLTK in python console:
import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

import os



def process_CV_result(input_file, confidence_thresh=0.5, drop_duplicates=True,  split_spaces=True, 
                        lowercase=True, remove_stops=True, remove_punctuation=True):

    with open(input_file, 'r', encoding='utf-8') as file:
        cv_result = json.load(file)

    if len(cv_result) == 0:
        print(f"Empty file found: {input_file}")
        return None

    all_column_headings = []
    all_flat_outputs = []

    useful_instagram_keys = ['account','profile_name','id','following','likes','posts_count','followers', 'image_url',
                            'biography','is_verified', 'datetime','comments','url','video_url','video_view_count','timestamp','caption']
    #'caption' ?

    for post in cv_result:

        column_headings = []
        flat_output = []

        for key, val in post['instagram'].items():
            if key in useful_instagram_keys:

                if key == 'is_verified' and val == '':
                    val = 0

                flat_output.append(val)
                column_headings.append(key)

        full_description = []
        for key, val in post['cvResult'].items():

            if key == 'imageType':
                for img_type, bool in val.items():
                    column_headings.append(img_type)
                    flat_output.append(bool)

            if key == 'color':
                for color_cat, color in val.items():
                    if color_cat == 'isBWImg':
                        column_headings.append(color_cat)
                        if color:
                            flat_output.append(1)
                        else:
                            flat_output.append(0)
                    if color_cat == 'dominantColors':   #Could we pass these to a color embedding ? 
                        for c in color:
                            full_description.append(c)

            if key == 'adult':
                for adult_cat, score in val.items():
                    if adult_cat in ['adultScore', 'racyScore', 'goreScore']:
                        column_headings.append(adult_cat)
                        flat_output.append(score)

            if key == 'tags':
                for tag in val:
                    if tag['confidence'] > confidence_thresh:
                        full_description.append(tag['name'])
            
            if key == 'description':
                [full_description.append(word) for word in val['tags']]
                all_confidences = [sentence['confidence'] for sentence in val['captions']]
                #Find the most confident caption
                caption = val['captions'][np.argmax(all_confidences)]
                #Only append it if it's greater than the confidence threshold
                if caption['confidence'] > confidence_thresh:
                    words = caption['text'].split(' ')
                    [full_description.append(word) for word in words]

            if key == 'objects':
                for object in val:
                    if object['confidence'] > confidence_thresh:
                        full_description.append(object['object'])

            #TODO: Not implemented faces, brands or categories yet?


        #Split words
        if split_spaces:
            temp_desc = []
            for word in full_description:
                [temp_desc.append(s) for s in word.split(' ')]
            full_description = temp_desc
        
        '''
        #Lowercase
        if lowercase:
            full_description = [word.lower() for word in full_description]
        
        #Stop words
        if remove_stops:
            #Remove stop words
            custom_stopwords = ['a', 'of', 'to', 'next', 'and']
            my_stop_words = pp.STOPWORDS.union(set(custom_stopwords))
            full_description = pp.remove_stopword_tokens(full_description, stopwords=my_stop_words)

        #Lemmatize
        if lemmatizer is not None:
            full_description = [lemmatizer.lemmatize(word) for word in full_description]

        #Remove apostrophes (acts as a double check on the lemmatizer as WordNet sucks)
        if remove_punctuation:
            full_description = [word.split("'")[0] for word in full_description]
            full_description = [word.replace(",", "") for word in full_description]

        #Drop duplicates
        if drop_duplicates:
            full_description = list(dict.fromkeys(full_description))

        '''


        column_headings.append('tokens')
        flat_output.append(full_description)
        all_column_headings.append(column_headings)
        all_flat_outputs.append(flat_output)


    output_df = pd.DataFrame(all_flat_outputs, columns=all_column_headings[0])
    return output_df


if __name__ == '__main__':

    
    #Example usage: 
    outfile = "data/postprocessed/NLP_project_kickoff_data.csv"
    data_path = 'data/json'
    output_df = None
    lemmatizer = WordNetLemmatizer()
    files = os.listdir(data_path)
    for file in files:

        input = data_path + '/' + file
        output = process_CV_result(input, lemmatizer=lemmatizer)
        if output_df is None:
            output_df = output
        else:
            output_df = pd.concat([output_df, output], axis=0)
    output_df.to_csv(outfile)
