import requests
import json
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import os
from os import listdir

WIKINEWS_URL = 'https://en.wikinews.org/w/api.php'
WIKINEWS_ALLPAGES = {
    'action': 'query',
    'format': 'json',
    'list': 'allpages',
    'aplimit': 'max'
}
WIKINEWS_CONTENT = {
    'action': 'parse',
    'prop': 'text',
    'formatversion': '2',
    'format': 'json'
}

def normalize(title):
    return title.replace(' ', '_')

def passages_from_soup(soup):
    date = ""
    passages = []
    r = re.compile(r"(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day, (January|February|March|April|May|June|July|August|September|October|November|December) \d+, \d\d\d\d") 
    
    for i, p in enumerate(soup.find_all("p")):
        text = p.get_text()

        if i == 0:
            match = r.search(text)
            if match != None:
                date = text[match.span()[0]:match.span()[1]]
                text = text[:match.span()[0]] + text[match.span()[1]:]

        if "Share this:" in text or "Share it!" in text:
            break

        if len(text) < 5:
            continue

        passages.append(text)
        
    if len(passages) < 5:
        return '', []
    
    return date, passages

def api_request(url, **kwargs):
    
    # create request url
    url += '?'
    for key, value in kwargs.items():
        if value == None:
            continue
        url += '{key}={value}&'.format(key=key, value=value)
    url = url[:-1]
    
    # make request
    response = requests.get(url)
    if response.status_code != 200:
        return '', False
    return response.content, True

def get_titles(num):

    titles = []
    wikinews_allpages = WIKINEWS_ALLPAGES

    while len(titles) <= num:
        
        content, success = api_request(WIKINEWS_URL, **wikinews_allpages)
        if not success:
            print('WARNING: Api request failed')
        content_json = json.loads(content)

        for page in content_json['query']['allpages']:
            titles.append(page['title'])

        try:
            wikinews_allpages['apcontinue'] = content_json['continue']['apcontinue']
        except:
            break

    return titles[:num]

def get_content(title, recursion=False):
    
    wikinews_content = WIKINEWS_CONTENT
    wikinews_content['page'] = normalize(title)

    content, success = api_request(WIKINEWS_URL, **wikinews_content)
    if not success:
        print('WARNING: Api request failed')

    content_json = json.loads(content)
    
    try:
        html = content_json['parse']['text']
    except:
        print('Error in content: ', content_json)
        return {"title": title, "date": "", "passages": []}

    soup = BeautifulSoup(html, 'html.parser')
    
    if soup.find("div", {'class': "redirectMsg"}) != None:
        if recursion == True:
            return {"title": title, "date": "", "passages": []}
        title = soup.find("div", {'class': "redirectMsg"}).a['title']
        return get_content(title, recursion=True)
    
    date, passages = passages_from_soup(soup)
    return {"title": title, "date": date, "passages": passages}

def create_data_wikinews(num, output):
    
    # get titles
    titles = get_titles(num)

    # write content
    with open(output, 'w') as f:
        for title in tqdm(titles):
            content = get_content(title)
            if len(content['passages']) == 0:
                continue
            json.dump(content, f)
            f.write('\n')

    # remove duplicate lines
    os.system('uniq {} uniq.{}'.format(output, output))
    os.system('rm {}'.format(output))
    os.system('mv uniq.{} {}'.format(output, output))

def create_formated_data(input, output):
    
    new_doc = 'Article\t{id}\n'
    new_sentence = 'Sentence\t{id}\t{text}\t{gold}\n'
    
    with open(output, 'w') as output_f:
        with open(input, 'r') as input_f:

            doc_id = 0
            for line in tqdm(input_f, desc="Reading train file"):

                doc_id += 1

                # Read
                json_line = json.loads(line)
                passages = json_line['passages']
                X_ = []
                y_ = []

                for passage in passages:
                    sentences = sent_tokenize(passage)
                    X_ += sentences
                    y_ += [1]+[0]*(len(sentences)-1)

                # Write
                output_f.write(new_doc.format(id=doc_id))
                for sentence_id, (sentence, gold) in enumerate(zip(X_, y_)):
                    output_f.write(new_sentence.format(id=sentence_id+1, text=sentence.replace('\n', ''), gold=gold))
                output_f.write('\n')

def split_data(input, train_output, test_data_output, test_gold_output, test_size=0.2, random_state=2020):
    
    # Read
    with open(input, 'r') as f:
        data = f.read()
        articles = data.split('\n\n')[:-1]
    
    train_articles, test_articles = train_test_split(articles, random_state=random_state, test_size=test_size)
    
    # Write train
    with open(train_output, 'w') as f:
        f.write('\n\n'.join(train_articles))

    # Write test
    with open(test_data_output, 'w') as data_f:
        with open(test_gold_output, 'w') as gold_f:
            for article in test_articles:
                lines = article.split('\n')
                data_f.write(lines[0]+'\n')
                gold_f.write(lines[0]+'\n')
                for line in lines[1:]:
                    elems = line.split('\t')
                    data_f.write('\t'.join(elems[:-1])+'\n')
                    gold_f.write('\t'.join(elems[:-2]+elems[-1:])+'\n')
                data_f.write('\n')
                gold_f.write('\n')

def create_train_test_data(input, train_output, test_data_output, test_gold_output, test_size=0.2, random_state=2020):
    output = 'data.txt'
    create_formated_data(input, output)
    split_data(output, train_output, test_data_output, test_gold_output, test_size, random_state)
    os.system('rm {}'.format(output))

''' Many functions since the part of utils.py are from: https://machinelearningmastery.com/prepare-news-articles-text-summarization/ '''

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load all stories in a directory
def load_stories(directory):
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)

# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

# load all stories in a directory
def load_stories(directory):
	all_stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		all_stories.append({'story':story, 'highlights':highlights})
	return all_stories

def get_passages(story):
    lines = story.split('\n\n')
    passages = []
    for line in lines:
        index = line.find('(CNN)  -- ')
        if index > -1:
            line = line[index+len('(CNN)  -- '):]
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN) -- '):]
        index = line.find('(CNN)')
        if index > -1:
            line = line[index+len('(CNN)'):]
        if len(line) > 0:
            passages.append(line)
    return passages

def create_data_cnn(directory, output):
    stories = load_stories(directory)
    # write content
    with open(output, 'w') as f:
        for story in stories:
            passages = get_passages(story['story'])
            content = {"title": 'title', "date": 'date', "passages": passages}
            if len(content['passages']) == 0:
                continue
            json.dump(content, f)
            f.write('\n')