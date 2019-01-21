import json, codecs, random, pickle, traceback, logging, os, math, time

def load_wikidata(dir):
    start = time.time()
    with codecs.open(dir+'/wikidata.json','r','utf-8') as data_file:
        wikidata = json.load(data_file)

    with codecs.open(dir+'/wikidata_rev.json','r','utf-8') as data_file:
        reverse_dict = json.load(data_file)
   
    with codecs.open(dir+'/wikidata_ent_types.json','r','utf-8') as data_file:
        wikidata_ent_types = json.load(data_file)
        
    with codecs.open(dir+'/wikidata_rev_ent_types.json','r','utf-8') as data_file:
        wikidata_rev_ent_types = json.load(data_file)
             
    with codecs.open(dir+'/wikidata_types.json','r','utf-8') as data_file:         
        wikidata_types = json.load(data_file)

    with codecs.open(dir+'/wikidata_rev_types.json','r','utf-8') as data_file:
        wikidata_rev_types = json.load(data_file)

    end = time.time()
    print 'Time taken to load wikidata ', (end-start), 'seconds'    
    
    return wikidata, reverse_dict, wikidata_types, wikidata_rev_types, wikidata_ent_types, wikidata_rev_ent_types


