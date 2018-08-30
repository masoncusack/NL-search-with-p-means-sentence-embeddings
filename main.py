import json
import logging
import os
import sys

import click
import numpy as np
from flask import Flask, render_template, jsonify
from flask import request
import pandas as pd
import string, re
import cPickle as pickle

from sentence_embeddings import get_sentence_embedding, operations
from word_embeddings import embeddings

from azure.storage.blob import BlockBlobService
import os
import json
from flask_cors import CORS

#Specify embedding types to use - for more examples or to use multiple embedding types at the same time, 
#see https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings

chosen_embedding_types = [('glove', ['p_mean_3'])]

def start_webserver(model_name, embeddings, host, port, logger):

    embedding_types_dict = dict([(e.alias, e) for e in embeddings])

    app = Flask('concatenated-p-means-embeddings-{}'.format(model_name))
    CORS(app)

    #Load in pickled dict of data
    with open('./pmeansembeddings.pickle', 'rb') as pickle_file:

        #if loading this pickle into a python3 script, remember param encoding="latin1"
        #nomenclature if your industry-specific information, stored in an Excel file
        nomenclature_dict = pickle.load(pickle_file)

    #cosine similarity helper function for embeddings-based search
    def cosine_similarity(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    #generate embedding from user input for search (internally - not returning to caller)
    def get_input_embedding(input_sent):
        tokens = input_sent.split()
        for embedding_type_name, operation_names in chosen_embedding_types:
            embeddings = embedding_types_dict[embedding_type_name]
            result_embedding = get_sentence_embedding(
                tokens,
                embeddings,
                operation_names
            )
        return result_embedding

    @app.route("/")
    def index():
        #render standard p-means homepage
        return render_template(
            'index.html',
            model_name=model_name, embeddings=embeddings, operations=operations.keys(), host=request.host
        )

    #search endpoint
    @app.route("/search", methods=['GET'])
    def search(N=100):

        try:
            search_term = request.args.get('q')
            print("input = " + search_term)
        
        except Exception as e:
            raise e

        if not search_term:
            return bad_request()
        else:
            this_embedding = get_input_embedding(search_term)

            #dictionary mapping search results to their cosine similarity value with the search input
            results_and_cosines = {}
            
            for key, value in nomenclature_dict.iteritems():
                #print("cosine = " + str(cosine_similarity(this_embedding, value)))
                similarity = cosine_similarity(this_embedding, value)
                #key = search result
                results_and_cosines.update({float(similarity): key})

            #sort values
            sort_vals = list(reversed(sorted(results_and_cosines.keys())))[:N] #reverse and get highest N values
            #use vals as lookup for keys (results) 
            #--> higher the similarity between search term and our data, the higher that result ranks in response
            top_results = [results_and_cosines[val] for val in sort_vals]
            #zip search results together with cosine similarity scores
            final_results = dict(zip(top_results, sort_vals))

            return jsonify(results)

    @app.route("/embed", methods=['POST'])
    def convert():
        try:
            output = ''
            conversion_data = json.loads(request.form.get('conversion'))

            sentences = conversion_data['sentences']
            print("\nSentences:")
            print(sentences)
            print("\n")
            chosen_embedding_types = conversion_data['embedding_types']
            # [('glove', ['mean'], ...), ...

            for i, sentence in enumerate(sentences):
                embs = []
                tokens = sentence.split()
                tokens_lower = sentence.lower().split()
                for embedding_type_name, operation_names in chosen_embedding_types:
                    embeddings = embedding_types_dict[embedding_type_name]
                    embs.append(get_sentence_embedding(
                        tokens_lower if embeddings.lowercased else tokens,
                        embeddings,
                        operation_names
                    ))
                concat_emb = np.concatenate(embs, axis=0)
                output += ' '.join([str(e) for e in concat_emb]) + '\n'
            return output
        except BaseException as e:
            logger.exception('Error while processing sentence embedding conversion request')
            return 'There was an error while processing sentence embedding conversion request (logged). \n' \
                   + 'Usually this is related to malformed json payload.', 500
    
    print('Starting server...')
    app.run(host=host, port=port, debug=False)

#Returns list of embeddings from dictionary of {result, description}
#keys = keys of original dictionary fed in
def convert_from_dict(dict_to_convert, word_embeddings):
    embedding_types_dict = dict([(e.alias, e) for e in word_embeddings])

    try:
        output = ''

        sentences = []

        for sentence in dict_to_convert.values():
            formatted_sentence = sentence.lower()
            sentences.append(re.sub(r'[^\d+\w\s]', '', formatted_sentence.encode('utf-8')))

        print("Sentence 1:" + sentences[0])
        print("Final sentence: " + sentences[len(sentences)-1])
        print("Total sentences: " + str(len(sentences)))

        #chosen_embedding_types is global
        # Format: [('glove', ['mean'], ...), ...

        embs = []
        for i, sentence in enumerate(sentences):
            tokens = sentence.split()
            for embedding_type_name, operation_names in chosen_embedding_types:
                embeddings = embedding_types_dict[embedding_type_name]
                embs.append(get_sentence_embedding(
                    tokens,
                    embeddings,
                    operation_names
                ))
            #concat_emb = np.concatenate(embs, axis=0)

        #Clean up  
        #TODO: extend word embeddings corpus to further prevent conversion failures 
        print("Removing failures from embeddings...")
        #create dictionary mapping results (keys) to description embeddings
        result_to_embed_dict = dict(zip(dict_to_convert.keys(), embs))
        print("result_to_embed_dict created fine...")
        for key, value in result_to_embed_dict.iteritems():
            #if we can't compute sentence embedding
            if "No word embeddings for sentence" in str(value):
                del result_to_embed_dict[key]
        
        print("Done! Pickling embeddings...")

        #Write to pickle if not already available   
        if not os.path.isfile("./pmeansembeddings.pickle"):
            with open('./pmeansembeddings.pickle', 'wb') as result_to_embed_pkl:
                pickle.dump(result_to_embed_dict, result_to_embed_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                print("Done! Check local file system.")
        else:
            print("pmeansembeddings.pickle already exists!")

        embs = result_to_embed_dict.values() #list consisting only of successful embeddings
        return embs
    except BaseException as e:
        print('\ERROR while processing sentence embedding conversion request:\n' + str(e) +"\n")
        #return
        

@click.command() #monolingual embedding default, though this can be changed
@click.option('--model', default='monolingual', help='en-de, en-fr, or monolingual')
@click.option('--embeddings-folder', default='data', help='path where the word embeddings will be stored')
@click.option('--webserver-host', default='0.0.0.0', help='For private (host-only) access set this to "127.0.0.1"')
@click.option('--webserver-port', default=5000, help='port of the webserver')

def run(model, embeddings_folder, webserver_host, webserver_port):
    logger = logging.getLogger('xling_sentence_embeddings')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel('DEBUG')
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)
    logger.setLevel('DEBUG')

    if not os.path.exists(embeddings_folder):
        os.mkdir(embeddings_folder)

    for emb in embeddings[model]:
        emb.download_file(embeddings_folder, logger)
    for emb in embeddings[model]:
        emb.load_vectors(embeddings_folder, logger)

    #load in secrets (populate your own secrets with the schema given in secrets_dev.json)
    with open('./secrets.json') as f:
        secrets = json.load(f)

    #Download nomenclature data for your particular industry from a blob store
    #This should contain potential results mapped to their descriptions in plain text
    block_blob_service = BlockBlobService(account_name=secrets["STORAGE_ACCOUNT_NAME"], account_key=secrets["STORAGE_ACCOUNT_KEY"]) 

    container_name ='nomenclature'

    generator = block_blob_service.list_blobs(container_name)
    for blob in generator:
        logger.info("\t Blob found: " + blob.name)

    #example excel file
    local_file_name ='Nomenclature.xlsx'

    path_to_file = './Nomenclature.xlsx'
    logger.info("\nDownloading excel to " + path_to_file)
    block_blob_service.get_blob_to_path(container_name, local_file_name, path_to_file)
    logger.info("Downloaded!")

    #pull results and descriptions into pandas dictionary with {key:colA, value:colG}
    #by 'results' we mean things your search will return, suggestions as a selection from your mass of data entries
    #the 'descriptions' will be used to generate embeddings we search over to return the associated results
    dataf = pd.read_excel('Nomenclature.xlsx', usecols="A, G")
    firstcol = "Result" #column 1 to add to dictionary (key)
    second_col = "Description" #column 2 to add to dictionary (value)
    #preprocess keys
    processed_keys = []
    for entry in dataf[firstcol]:
        head, tail = entry.split(" ") #remove everything after space
        processed = head
        processed_keys.append(processed)
    datadict = dict(zip(processed_keys, dataf[second_col]))

    logger.info("Nomenclature loaded info df and mapped to dictionary with {Key, value}.")

    #get embeddings from descriptions for search data, to search over given user input later
    logger.info("Generating description embeddings...")
    description_embeddings = convert_from_dict(datadict, embeddings[model])
    logger.info("Got " + str(len(description_embeddings)) + " embeddings for descriptions."+"\n")
    logger.info("Here's the first:\n")
    print(description_embeddings[0])

    #Done
    logger.info('Finished loading all data. The model is ready!')
    logger.info('Now starting the webserver (open in browser for more information!)...')

    start_webserver(model, embeddings[model], webserver_host, webserver_port, logger)

if __name__ == '__main__':
    run()
