

import json 
import pandas as pd 


json_file = "data/enriched_instagram.com_collector_20211106_161542..json"

with open(json_file, 'r', encoding='utf-8') as file:
    example = json.load(file)

test_examples = example[:50]



useful_instagram_keys = [
'account',
#'caption',
'profile_name',     #This will need encoding to an ID (will help with plotting from the same user) or removing
#'biography',
'id',
'following',
'likes',
'posts_count',
'followers',
'is_verified',
'datetime',
'comments',
'url',                  #Useful for checking the results by eye but not for training on
'video_url',            #Suggest encoding this into an image or video bool 
'video_view_count',        
'timestamp']
#print(test_example.keys())


all_column_headings = []
all_flat_outputs = []

for test_example in test_examples:


    for key, val in test_example['cvResult'].items():
        #print(key , ' -- ', val)
        ...
    #print('\n ---------- \n')

    column_headings = []
    flat_output = []

    for key, val in test_example['instagram'].items():
        if key in useful_instagram_keys:
            flat_output.append(val)
            column_headings.append(key)



    confidence_thresh = 0.3
    full_description = []
    for key, val in test_example['cvResult'].items():

        if key == 'imageType':
            for type, bool in val.items():
                column_headings.append(type)
                flat_output.append(bool)

        if key == 'color':
            for color_cat, color in val.items():
                if color_cat == 'isBWImg':
                    column_headings.append(color_cat)
                    flat_output.append(color)
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
            for sentence in val['captions']:
                if sentence['confidence'] > confidence_thresh:
                    words = sentence['text'].split(' ')
                    [full_description.append(word) for word in words]

        if key == 'objects':
            for object in val:
                if object['confidence'] > confidence_thresh:
                    full_description.append(object['object'])

    column_headings.append('tokens')
    flat_output.append(full_description)

    all_column_headings.append(column_headings)
    all_flat_outputs.append(flat_output)

    for heading, value in zip(column_headings, flat_output):
        #print(heading, " : " , value)
        ...

#print(test_example['instagram']['url'])

test = pd.DataFrame(all_flat_outputs, columns=all_column_headings[0])
test.to_csv('data/test_postprocessed_data.csv')
print(test.head())


easy_cvResult_keys = [
    'imageType',
    'adult',
]

useful_cvResult_keys = [
'categories',
'adult',
'color',
'imageType',
'tags',
'description',
'faces',
'objects',
'brands'
]

''' #Take the ones that can be flattened directly and combine the others into a list of tokens using a lower threshold
flatten_mapping = {
    'categories' : '',
'adult' : ['All'],   #Want to flatten to use scores?
'color' : ['All'], #Take the dominant color and use words
'imageType' : ['All'], #Flatten all as flags
'tags' ,
'description',
'faces',
'objects',
'brands'}
'''
