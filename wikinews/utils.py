import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import os

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

def create_data(num, output):
    
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
